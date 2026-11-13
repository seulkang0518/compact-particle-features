"""Export already-scaled ImageNet VAE latents into a bounded portable packet.

Generated NPZ files require latents [N,32,32,4], integer labels [N], string/integer
noise_ids [N], and a scalar string checkpoint_id. Optional cfg is scalar or [N].
No rescaling, VAE encoding, checkpoint loading, or image download is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from compact_features.data import identity_split, file_hash, write_json, check_capacity, verify_latents


def npz_latent_header(path):
    with zipfile.ZipFile(path) as archive, archive.open('latents.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1,0)
                            else np.lib.format.read_array_header_2_0(stream))
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def identifiers(values, size, name):
    value = np.asarray(values)
    if value.shape != (size,) or value.dtype.kind not in 'iuUS':
        raise ValueError(f'{name}: expected {size} integer or string IDs (object pickle arrays are not accepted)')
    result=[x.decode('utf8') if isinstance(x,bytes) else str(x) for x in value.tolist()]
    if any(not x for x in result):
        raise ValueError(f'{name}: empty identity')
    return result


def create_packet(args):
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError('Packet output exists; choose a new directory')
    if min(args.max_real_per_class, args.max_generated_per_class) < 1:
        raise ValueError('Positive per-class caps required')
    if not np.isfinite(args.max_input_gb) or args.max_input_gb<=0:
        raise ValueError('Positive per-file --max-input-gb cap required')
    real_path, labels_path = Path(args.real_latents).resolve(), Path(args.real_labels).resolve()
    real = np.load(real_path, mmap_mode='r', allow_pickle=False)
    labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
    verify_latents(real, labels, 'real')
    real_ids = identifiers(np.load(args.real_ids, allow_pickle=False), len(real), 'real_ids') if args.real_ids else [str(i) for i in range(len(real))]
    candidates = []
    for i, (label, identity) in enumerate(zip(labels, real_ids)):
        candidates.append({'kind':'real', 'split':identity_split('real:'+identity, args.seed),
            'class_id':int(label), 'source_id':'real:'+identity, 'source_index':i, 'origin':str(real_path)})
    sources = [{'kind':'real', 'latents':str(real_path), 'latents_sha256':file_hash(real_path),
                'labels_sha256':file_hash(labels_path)}]
    if args.real_ids:
        sources[0]['real_ids_sha256'] = file_hash(args.real_ids)
    seen, checkpoints = set(), set()
    for filename in args.generated:
        path = Path(filename).resolve()
        header = npz_latent_header(path)
        if np.prod(header.shape, dtype=np.int64) * header.dtype.itemsize > args.max_input_gb * 1024**3:
            raise ValueError(f'{path}: latent array exceeds the per-file --max-input-gb memory cap')
        with np.load(path, allow_pickle=False) as packet:
            labels = packet['labels']
            verify_latents(header, labels, str(path))
            checkpoint_value = packet['checkpoint_id']
            if checkpoint_value.ndim != 0 or checkpoint_value.dtype.kind not in 'US':
                raise ValueError('checkpoint_id must be one scalar string per generated NPZ file')
            checkpoint = checkpoint_value.item()
            checkpoint = checkpoint.decode('utf8') if isinstance(checkpoint,bytes) else str(checkpoint)
            if not checkpoint:
                raise ValueError('Empty checkpoint_id')
            checkpoints.add(checkpoint)
            noises = identifiers(packet['noise_ids'], len(labels), 'noise_ids')
            cfg = np.broadcast_to(np.asarray(packet['cfg'] if 'cfg' in packet else 1., dtype=np.float64), (len(labels),))
            if not np.isfinite(cfg).all():
                raise ValueError('CFG values must be finite')
            for i, (label, noise, guidance) in enumerate(zip(labels, noises, cfg)):
                identity = (checkpoint, int(label), noise, float(guidance))
                if identity in seen:
                    raise ValueError(f'Duplicate generated checkpoint/class/noise/CFG identity: {identity}')
                seen.add(identity)
                split = identity_split('noise:'+noise, args.seed)
                # A held-out checkpoint is not allowed to leak either its weights
                # or its noise identities into the training/validation packet.
                if (checkpoint == args.heldout_checkpoint) != (split == 'audit'):
                    continue
                candidates.append({'kind':'generated', 'split':split, 'class_id':int(label),
                    'source_id':f'generated:{checkpoint}:{label}:{noise}:{float(guidance):.17g}',
                    'noise_id':noise, 'checkpoint_id':checkpoint, 'cfg':float(guidance),
                    'source_index':i, 'origin':str(path)})
        sources.append({'kind':'generated', 'path':str(path), 'sha256':file_hash(path), 'checkpoint_id':checkpoint})
    if args.heldout_checkpoint not in checkpoints or len(checkpoints) < 2:
        raise ValueError('Supply the held-out checkpoint and at least one different training checkpoint')
    groups = {}
    for row in candidates:
        groups.setdefault((row['kind'],row['split'],row['class_id']), []).append(row)
    rows = []
    for (kind, _, _), group in sorted(groups.items()):
        cap = args.max_real_per_class if kind == 'real' else args.max_generated_per_class
        group.sort(key=lambda row: hashlib.sha256(f'{args.seed}:{row["source_id"]}:{row["source_index"]}'.encode()).hexdigest())
        rows.extend(group[:cap])
    counts = {split:{kind:sum(r['split']==split and r['kind']==kind for r in rows)
                    for kind in ('real','generated')} for split in ('train','val','audit')}
    if any(n == 0 for split in counts.values() for n in split.values()):
        raise ValueError(f'An empty source/split remains after identity hold-outs: {counts}; provide more identities')
    per_class={split:{} for split in counts}
    for row in rows:
        pool=per_class[row['split']].setdefault(str(row['class_id']),{'real':0,'generated':0})
        pool[row['kind']]+=1
    query_count,reference_count=getattr(args,'queries',8),getattr(args,'references',16)
    if query_count<2 or reference_count<1:
        raise ValueError('Eligibility requires --queries >=2 and --references >=1')
    eligibility={'queries':query_count,'references':reference_count,
        'classes':{split:sorted(int(c) for c,pool in classes.items()
                    if pool['real']>=reference_count and pool['generated']>=query_count)
                   for split,classes in per_class.items()}}
    print(json.dumps({'preflight_counts':counts,'eligibility':eligibility,
        'eligible_class_counts':{s:len(v) for s,v in eligibility['classes'].items()}},indent=2),flush=True)
    check_capacity(root, len(rows)*4*32*32*4 + len(rows)*768, args.max_packet_gb)
    # Avoid nesting a materialized output inside any input directory or vice versa.
    for path in [real_path, labels_path, *map(Path,args.generated)]:
        if path.resolve().is_relative_to(root):
            raise ValueError('Packet output contains an input file')
    root.mkdir(parents=True)
    manifest = {'version':1, 'complete':False, 'synthetic':False, 'layout':'BCHW',
        'state':'sampled SD-VAE latent, already multiplied by 0.18215; no additional scaling',
        'seed':args.seed, 'heldout_checkpoint':args.heldout_checkpoint,
        'split_policy':'70/15/15 deterministic original-real-ID / global-noise-ID hash; checkpoint held out for audit only',
        'sources':sources, 'counts':counts, 'per_class_counts':per_class,'eligibility':eligibility,
        'records':len(rows), 'files_sha256':{}}
    write_json(root/'manifest.json', manifest)
    output = np.lib.format.open_memmap(root/'inputs.npy', mode='w+', dtype='float32', shape=(len(rows),4,32,32))
    origins = {}
    for output_index, row in enumerate(rows):
        origins.setdefault(row['origin'], []).append((output_index,row['source_index']))
    for origin, selection in origins.items():
        holder = np.load(origin, mmap_mode='r', allow_pickle=False)
        values = holder if origin == str(real_path) else holder['latents']
        for start in range(0,len(selection),128):
            batch = selection[start:start+128]
            data = np.asarray(values[[i for _,i in batch]],dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(f'Nonfinite selected latent in {origin}; incomplete packet is deliberately retained')
            output[[i for i,_ in batch]] = data.transpose(0,3,1,2)
        if hasattr(holder,'close'):
            holder.close()
        print(f'Exported {len(selection)} scaled latents from {Path(origin).name}',flush=True)
    output.flush()
    del output
    write_json(root/'records.json',rows)
    manifest.update(complete=True, files_sha256={name:file_hash(root/name) for name in ('inputs.npy','records.json')})
    write_json(root/'manifest.json',manifest)
    print(json.dumps({'output':str(root),'counts':counts,'heldout_checkpoint':args.heldout_checkpoint},indent=2))
    return manifest


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-latents',required=True)
    p.add_argument('--real-labels',required=True)
    p.add_argument('--real-ids',help='Optional original-image IDs; all views of one image must share an ID')
    p.add_argument('--generated',nargs='+',required=True)
    p.add_argument('--heldout-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-real-per-class',type=int,default=128,help='Cap per class AND split')
    p.add_argument('--max-generated-per-class',type=int,default=128,help='Cap per class AND split across checkpoints')
    p.add_argument('--queries',type=int,default=8,help='Generated query count for the preflight eligibility report')
    p.add_argument('--references',type=int,default=16,help='Real reference count for the preflight eligibility report')
    p.add_argument('--max-packet-gb',type=float,default=4.)
    p.add_argument('--max-input-gb',type=float,default=4.,help='Maximum uncompressed generated latent array per NPZ (GiB)')
    return p.parse_args()


if __name__=='__main__':
    create_packet(parse_args())
