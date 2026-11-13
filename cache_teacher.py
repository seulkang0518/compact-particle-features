"""Cache fixed same-class comparison episodes for compact encoder research.

Real mode reads a portable packet and a LOCAL latent-MAE artifact. Synthetic
mode uses fixed random convolutions and is exclusively a pipeline check.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from compact_features.data import (STAGES, Packet, check_capacity, file_hash,
                                   collect_cached_features, write_json, spatial_position_pools)
from compact_features.teacher import MAETeacher, SyntheticTeacher


def synthetic_episode(split, number, queries, references, seed):
    # Explicit disjoint split seeds; no synthetic sample pretends to be ImageNet.
    split_id={'train':0,'val':1,'audit':2}[split]
    rng=np.random.default_rng(seed+100000*split_id+number)
    class_id=number%4
    values=rng.normal(0,.4,size=(queries+references,4,32,32)).astype('float32')
    grid=np.linspace(-1,1,32,dtype='float32')
    values += np.sin((class_id+1)*grid)[None,None,None,:]
    values[:queries] += .15
    metadata=[{'kind':'generated' if i<queries else 'real','class_id':class_id,
        'source_id':f'synthetic:{split}:{number}:{i}', 'split':split,
        **({'noise_id':f'{split}:{number}:{i}', 'checkpoint_id':'synthetic_heldout' if split=='audit' else 'synthetic_train',
            'cfg':1.} if i<queries else {})} for i in range(len(values))]
    return torch.from_numpy(values),class_id,metadata


def build_cache(args):
    wall_start=time.perf_counter()
    root=Path(args.output).resolve()
    if root.exists():
        raise FileExistsError('Cache output exists; choose a fresh directory')
    if min(args.queries,args.references,args.positions,args.microbatch,args.train_episodes,args.val_episodes,args.audit_episodes)<1:
        raise ValueError('Positive counts, positions and microbatch required')
    if args.queries<2:
        raise ValueError('At least two generated queries are required for repulsion')
    if args.synthetic and args.packet:
        raise ValueError('Use --synthetic OR --packet')
    if not args.synthetic and not (args.packet and args.source_repo and args.teacher_root):
        raise ValueError('Real mode requires --packet --source-repo --teacher-root')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.cpu_threads)
    packet=None if args.synthetic else Packet(args.packet)
    if packet and (root==packet.root or root.is_relative_to(packet.root) or packet.root.is_relative_to(root)):
        raise ValueError('Cache and input packet must be separate, nonnested directories')
    split_counts={'train':args.train_episodes,'val':args.val_episodes,'audit':args.audit_episodes}
    eligible={} if packet is None else {s:packet.eligible(s,args.queries,args.references) for s in split_counts}
    if packet and any(not classes for classes in eligible.values()):
        raise ValueError(f'Every split needs same-class pools with Q={args.queries}, R={args.references}; eligible counts: '
                         f'{ {s:len(c) for s,c in eligible.items()} }')
    teacher_start=time.perf_counter()
    teacher=(SyntheticTeacher(args.synthetic_width,args.seed+1234) if args.synthetic else
             MAETeacher(args.source_repo,args.teacher_root,args.teacher_name)).to(device).eval()
    count=args.queries+args.references
    with torch.no_grad():
        example=teacher(torch.zeros((1,4,32,32),device=device))
    channels={stage:int(example[stage].shape[1]) for stage in STAGES}
    sizes={stage:list(example[stage].shape[-2:]) for stage in STAGES}
    if channels!=teacher.channels:
        raise ValueError('Teacher channel metadata disagrees with actual maps')
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    teacher_seconds=time.perf_counter()-teacher_start
    position_pools=spatial_position_pools(sizes,args.seed)
    if any(args.positions>len(pool[k]) for pool in position_pools.values() for k in ('development','audit')):
        raise ValueError('--positions exceeds a disjoint spatial half-pool; use at most half the smallest stage map')
    dtype=torch.float16 if args.cache_dtype=='float16' else torch.float32
    bytes_per_value=2 if dtype==torch.float16 else 4
    feature_values=sum((min(args.positions,np.prod(sizes[s]))+2)*channels[s]*count for s in STAGES)
    estimated_episode=int(feature_values*bytes_per_value + count*4*32*32*4 + 16384)
    required=check_capacity(root,estimated_episode*sum(split_counts.values()),args.max_cache_gb)
    print(json.dumps({'estimated_cache_gib_with_margin':required/1024**3,'teacher_channels':channels,
        'feature_sizes':sizes,'synthetic':args.synthetic,'episodes':split_counts},indent=2),flush=True)
    root.mkdir(parents=True)
    (root/'episodes').mkdir()
    index={'version':1,'complete':False,'synthetic':args.synthetic,
        'protocol':'same-class Q generated queries followed by R real references; five common stages, sampled spatial positions + global mean/std',
        'teacher_channels':channels,'feature_sizes':sizes,'queries':args.queries,'references':args.references,
        'cache_dtype':args.cache_dtype,'seed':args.seed,'teacher':teacher.provenance,
        'position_policy':'fixed disjoint development(train/val) vs audit local-position pools; global mean/std use all positions',
        'position_pools':position_pools,'teacher_construction_and_shape_check_seconds':teacher_seconds,
        'eligible_classes':eligible,'packet_manifest_sha256':None if packet is None else file_hash(packet.root/'manifest.json'),
        'episodes':[],'estimated_output_bytes_with_margin':required}
    write_json(root/'index.json',index)
    global_number=0
    for split,episodes in split_counts.items():
        for number in range(episodes):
            episode_seed=args.seed + {'train':0,'val':100000,'audit':200000}[split]+number
            inputs,class_id,records=(synthetic_episode(split,number,args.queries,args.references,args.seed) if packet is None
                                    else packet.episode(split,args.queries,args.references,episode_seed))
            rng=np.random.default_rng(episode_seed+30933)
            pool_name='audit' if split=='audit' else 'development'
            positions={stage:torch.as_tensor(rng.choice(position_pools[stage][pool_name],
                            args.positions,replace=False),dtype=torch.long) for stage in STAGES}
            features={}
            with torch.no_grad():
                for start in range(0,len(inputs),args.microbatch):
                    maps=teacher(inputs[start:start+args.microbatch].to(device))
                    chunk=collect_cached_features(maps,positions)
                    for name,value in chunk.items():
                        features.setdefault(name,[]).append(value.to(device='cpu',dtype=dtype))
                    del maps,chunk
            features={name:torch.cat(chunks,dim=1) for name,chunks in features.items()}
            if not all(torch.isfinite(value).all() for value in features.values()):
                raise ValueError('Nonfinite teacher features; choose float32 cache dtype if half precision overflowed')
            relative=f'episodes/{global_number:06d}.pt'
            payload={'inputs':inputs.float(),'positions':positions,'teacher_features':features,
                'target_sizes':sizes,'queries':args.queries,
                'metadata':{'split':split,'class_id':class_id,'episode_seed':episode_seed,
                            'source_records':records,'synthetic':args.synthetic}}
            temporary=(root/relative).with_suffix('.tmp')
            torch.save(payload,temporary)
            temporary.replace(root/relative)
            index['episodes'].append({'file':relative,'split':split,'class_id':class_id,
                                     'queries':args.queries,'sha256':file_hash(root/relative)})
            global_number+=1
            if number==0 or (number+1)%20==0 or number+1==episodes:
                print(f'{split}: cached {number+1}/{episodes} episodes',flush=True)
    index['complete']=True
    index['cache_wall_seconds']=time.perf_counter()-wall_start
    index['episode_file_bytes']=sum((root/item['file']).stat().st_size for item in index['episodes'])
    write_json(root/'index.json',index)
    # A second write includes the index itself; the byte count converges once
    # the number of decimal digits is unchanged.
    for _ in range(3):
        actual=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if index.get('actual_cache_bytes')==actual:
            break
        index['actual_cache_bytes']=actual
        write_json(root/'index.json',index)
    print(f'Cache complete: {root}/index.json',flush=True)
    return index


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--packet')
    p.add_argument('--source-repo')
    p.add_argument('--teacher-root')
    p.add_argument('--teacher-name',default='mae_latent_640')
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--queries',type=int,default=8)
    p.add_argument('--references',type=int,default=16)
    p.add_argument('--positions',type=int,default=4)
    p.add_argument('--train-episodes',type=int,default=256)
    p.add_argument('--val-episodes',type=int,default=64)
    p.add_argument('--audit-episodes',type=int,default=64)
    p.add_argument('--synthetic-width',type=int,default=4)
    p.add_argument('--cache-dtype',choices=['float16','float32'],default='float16')
    p.add_argument('--max-cache-gb',type=float,default=4.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=2)
    return p.parse_args()


if __name__=='__main__':
    build_cache(parse_args())
