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

from compact_features.audit_math import perturb_latents, reference_scales, mmd2
from compact_features.runtime import write_json, digest


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
    args=p.parse_args()
    if min(args.queries,args.references)<2 or min(args.microbatch,args.repeats)<1:
        raise ValueError('Need >=2 queries/references, positive microbatch/repeats')
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
        encoder=MAETeacher(args.source_repo,args.teacher).cuda().eval()
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
    rows=[]
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
    baseline=rows[0]['mmd2']
    for row in rows: row['mmd2_minus_clean_control']=row['mmd2']-baseline
    write_json(out/'summary.json',{'encoder':args.encoder,'class_name':folder.name,'seed':args.seed,
        'queries':args.queries,'references':args.references,'microbatch':args.microbatch,'repeats':args.repeats,
        'reference_scales':{k:float(v) for k,v in scales.items()},'provenance':provenance,'results':rows,
        'selected_images':[{'path':str(files[int(i)]),'sha256':digest(files[int(i)])} for i in selected],
        'vae_config_sha256':digest(Path(args.vae)/'config.json'),'torch':torch.__version__,
        'device':torch.cuda.get_device_name(),
        'scope':'Detection/input-gradient-cost diagnostic only. MAE uses five global-mean branches, not full original recipe. Latent perturbations are not established perceptual rankings; raw MMDs across encoders are not comparable. No correction, FID, or generation-quality claim.'})


if __name__=='__main__': main()
