"""Small frozen-encoder diagnostic on common scaled VAE-latent inputs.

One class, disjoint real reference/query images; latent perturbations are probes,
not established quality rankings. This does not train a generator or an encoder.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compact_features.audit_math import (perturb_latents, reference_scales, mmd2,
                                        normalized_descent, recovery_metrics, paired_feature_errors)
from compact_features.runtime import write_json, digest, source_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', required=True, help='ImageNet TRAIN directory containing synset subdirectories')
    p.add_argument('--vae', required=True, help='Local diffusers VAE directory')
    p.add_argument('--teacher', required=True, help='Local MAE artifact directory')
    p.add_argument('--source-repo', required=True)
    p.add_argument('--encoder', choices=['mae','inception'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--class-name', help='Default: alphabetically first training synset')
    p.add_argument('--queries', type=int, default=4)
    p.add_argument('--references', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--correction-steps', type=int, default=0)
    p.add_argument('--correction-objective', choices=['mmd','paired_feature'], default='mmd',
                   help='Paired feature regression is a separate inversion sanity control, not MMD')
    p.add_argument('--step-rms', type=float, default=.01,
                   help='Per-image latent RMS update size; fixed across encoders')
    args=p.parse_args()
    if min(args.queries,args.references)<2 or min(args.microbatch,args.repeats)<1:
        raise ValueError('Need >=2 queries/references, positive microbatch/repeats')
    if args.correction_steps < 0 or args.step_rms <= 0:
        raise ValueError('Need nonnegative correction steps and positive step RMS')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this real-checkpoint audit inside an allocated GPU job')
    from PIL import Image
    from diffusers import AutoencoderKL
    from torchvision.utils import save_image
    torch.set_num_threads(4)
    out=Path(args.output)
    if out.exists():
        raise FileExistsError('Choose a fresh output directory')
    root=Path(args.image_root)
    classes=sorted(d for d in root.iterdir() if d.is_dir())
    folder=root/args.class_name if args.class_name else classes[0]
    files=sorted(f for f in folder.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'})
    g=np.random.default_rng(args.seed)
    selected=g.choice(len(files),args.queries+args.references,replace=False)
    images=[]
    for i in selected:
        with Image.open(files[int(i)]) as im:
            im=im.convert('RGB')
            while min(im.size)>=512:
                im=im.resize(tuple(d//2 for d in im.size),Image.Resampling.BOX)
            scale=256/min(im.size)
            im=im.resize(tuple(round(d*scale) for d in im.size),Image.Resampling.BICUBIC)
            a=np.asarray(im); y=(a.shape[0]-256)//2; x=(a.shape[1]-256)//2
            images.append(torch.from_numpy(a[y:y+256,x:x+256].copy()).permute(2,0,1).float()/127.5-1)
    vae=AutoencoderKL.from_pretrained(args.vae,local_files_only=True).eval().requires_grad_(False).cuda()
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    latents=[]
    with torch.no_grad():
        for image in images:
            latents.append(vae.encode(image[None].cuda()).latent_dist.sample(generator=rng)*.18215)
    states=torch.cat(latents)
    reference=states[args.queries:].detach(); clean=states[:args.queries].detach()
    out.mkdir(parents=True)
    cases={kind:perturb_latents(clean,kind,args.seed) for kind in ('clean','noise','blur','shuffle')}
    with torch.no_grad():
        for name,x in {'reference':reference,**cases}.items():
            rgb=torch.cat([vae.decode(chunk/.18215).sample for chunk in x.split(args.microbatch)])
            save_image((rgb.cpu()/2+.5).clamp(0,1),out/f'{name}.png',nrow=4)
    if args.encoder=='mae':
        del vae; gc.collect(); torch.cuda.empty_cache()
        from compact_features.teacher import MAETeacher
        encoder=MAETeacher(args.source_repo,args.teacher,force_fp32=True).cuda().eval()
        def features(x):
            return {k:v.float().mean((2,3)) for k,v in encoder(x).items()}
        provenance=encoder.provenance
    else:
        from torchvision.models import inception_v3, Inception_V3_Weights
        # Load pretrained weights on the login node first; GPU job can be offline.
        encoder=inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,transform_input=False)
        encoder.fc=nn.Identity(); encoder.aux_logits=False; encoder.AuxLogits=None
        encoder=encoder.cuda().eval().requires_grad_(False)
        def features(x):
            rgb=(vae.decode(x/.18215).sample+1)/2  # no clipping: retain gradients
            rgb=F.interpolate(rgb,size=(299,299),mode='bilinear',align_corners=False)
            mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
            std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
            return {'pooled':encoder((rgb-mean)/std)}
        provenance={'weights':'torchvision IMAGENET1K_V1','input':'VAE-decoded RGB, direct bilinear resize299, ImageNet normalization; no center crop',
                    'decoder_in_timing':True}
    def batch_features(x):
        chunks=[features(c) for c in x.split(args.microbatch)]
        return {k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    with torch.no_grad():
        ref=batch_features(reference)
        scales=reference_scales(ref)
        targets=batch_features(clean) if args.correction_steps and args.correction_objective=='paired_feature' else None
    rows=[]
    correction_rows=[]
    snapshots={}
    for name,x in cases.items():
        def measure():
            q=x.detach().requires_grad_(True)
            loss=mmd2(batch_features(q),ref,scales)
            grad,=torch.autograd.grad(loss,q)
            return float(loss.detach()),float(grad.norm().detach())
        measure()  # warmup
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(args.repeats): value,norm=measure()
        torch.cuda.synchronize()
        row={'case':name,'mmd2':value,'latent_gradient_l2':norm,
             'mean_forward_mmd_input_backward_ms':1000*(time.perf_counter()-start)/args.repeats,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        rows.append(row); print(json.dumps(row),flush=True)
        if args.correction_steps:
            q=x.detach()
            for step in range(args.correction_steps+1):
                q=q.detach().requires_grad_(True)
                q_features=batch_features(q)
                discrepancy=mmd2(q_features,ref,scales)
                errors=paired_feature_errors(q_features,targets,scales) if targets is not None else None
                loss=errors.mean() if errors is not None else discrepancy
                grad,=torch.autograd.grad(loss,q)
                # Do not amplify roundoff-sized paired residuals into a full
                # fixed-RMS update at an already matched clean target.
                if errors is not None:
                    active=errors.detach()>1e-10
                    grad=grad*active.reshape((-1,)+(1,)*(q.ndim-1))
                item={'case':name,'step':step,'mmd2':float(discrepancy.detach()),
                      'correction_objective':args.correction_objective,'objective_value':float(loss.detach()),
                      'displacement_rms':float((q.detach()-x).square().mean().sqrt()),
                      **recovery_metrics(q,clean,grad)}
                correction_rows.append(item)
                if step in {0,1,5,args.correction_steps}:
                    snapshots[f'{name}_step{step:03d}']=q.detach().cpu()
                    print(json.dumps({'correction':item}),flush=True)
                if step < args.correction_steps:
                    q=normalized_descent(q,grad,args.step_rms)
            write_json(out/'correction_trace.json',correction_rows)
            del q,grad,loss,discrepancy,q_features,errors
    baseline=rows[0]['mmd2']
    for row in rows: row['mmd2_minus_clean_control']=row['mmd2']-baseline
    if snapshots:
        # Reload decoder only after releasing the MAE to keep its cost outside
        # timing and avoid keeping both large models resident during the audit.
        if args.encoder=='mae':
            del encoder; gc.collect(); torch.cuda.empty_cache()
            vae=AutoencoderKL.from_pretrained(args.vae,local_files_only=True).eval().requires_grad_(False).cuda()
        with torch.no_grad():
            clean_rgb=torch.cat([vae.decode(c/.18215).sample for c in clean.split(args.microbatch)])
            for name,state in snapshots.items():
                rgb=torch.cat([vae.decode(c.cuda()/.18215).sample for c in state.split(args.microbatch)])
                save_image((rgb.cpu()/2+.5).clamp(0,1),out/f'{name}.png',nrow=4)
                case,step=name.rsplit('_step',1)
                for row in correction_rows:
                    if row['case']==case and row['step']==int(step):
                        row['paired_decoded_rgb_mse']=float(((rgb-clean_rgb)/2).square().mean())
        write_json(out/'correction_trace.json',correction_rows)
    write_json(out/'summary.json',{'encoder':args.encoder,'class_name':folder.name,'seed':args.seed,
        'queries':args.queries,'references':args.references,'microbatch':args.microbatch,'repeats':args.repeats,
        'reference_scales':{k:float(v) for k,v in scales.items()},'provenance':provenance,'results':rows,
        'selected_images':[{'path':str(files[int(i)]),'sha256':digest(files[int(i)])} for i in selected],
        'vae_config_sha256':digest(Path(args.vae)/'config.json'),'torch':torch.__version__,
        'device':torch.cuda.get_device_name(),
        'code_sha256':source_digest(),
        'correction_steps':args.correction_steps,'step_rms':args.step_rms,
        'correction_objective':args.correction_objective,
        'paired_feature_stop_tolerance':1e-10 if targets is not None else None,
        'correction_results':correction_rows,
        'scope':'Exploratory frozen-encoder diagnostic. MMD descent uses independent real references; paired_feature instead uses each query own clean features as its stopped regression target. Fixed-budget feature inversion need not reconstruct the original image. Pixel/latent recovery is not semantic quality; population matching need not recover each original. MAE uses global-mean branches, not full original recipe. No FID or generator-quality claim.'})


if __name__=='__main__': main()
