from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from compact_features.data import Packet, identity_split, STAGES
from prepare_packet import create_packet
from cache_teacher import build_cache


def packet_inputs(tmp_path):
    n=120
    values=np.arange(n,dtype='float32')[:,None,None,None]*np.ones((n,32,32,4),dtype='float32')
    np.save(tmp_path/'real.npy',values)
    np.save(tmp_path/'labels.npy',np.arange(n,dtype='int64')%2)
    noise=np.array([f'noise-{i}' for i in range(100)]*2)
    labels=np.repeat(np.arange(2),100).astype('int64')
    generated=[]
    for number,checkpoint in enumerate(('early','late','heldout')):
        name=tmp_path/f'{checkpoint}.npz'
        latent=np.full((200,32,32,4),.25+number,dtype='float32')
        np.savez(name,latents=latent,labels=labels,noise_ids=noise,
                 checkpoint_id=np.array(checkpoint),cfg=np.array(1.19))
        generated.append(str(name))
    return Namespace(output=str(tmp_path/'packet'),real_latents=str(tmp_path/'real.npy'),
        real_labels=str(tmp_path/'labels.npy'),real_ids=None,generated=generated,
        heldout_checkpoint='heldout',seed=42,max_real_per_class=100,
        max_generated_per_class=100,max_packet_gb=.2,max_input_gb=.2)


def test_packet_keeps_scaled_latents_and_holds_out_noise_and_checkpoint(tmp_path):
    args=packet_inputs(tmp_path)
    create_packet(args)
    packet=Packet(args.output)
    split_noises={s:set() for s in ('train','val','audit')}
    for i,row in enumerate(packet.records):
        if row['kind']=='generated':
            split_noises[row['split']].add(row['noise_id'])
            assert (row['checkpoint_id']=='heldout') == (row['split']=='audit')
        else:
            # These are already-scaled values: no additional 0.18215 factor.
            np.testing.assert_array_equal(packet.inputs[i],float(row['source_index']))
            assert row['split']==identity_split(row['source_id'],42)
    assert split_noises['train'].isdisjoint(split_noises['val'])
    assert split_noises['train'].isdisjoint(split_noises['audit'])
    assert split_noises['val'].isdisjoint(split_noises['audit'])
    inputs,label,records=packet.episode('audit',2,2,123)
    assert inputs.shape==(4,4,32,32)
    assert all(r['class_id']==label for r in records)
    assert [r['kind'] for r in records]==['generated','generated','real','real']
    assert len({r['source_id'] for r in records})==4
    again,_,_=packet.episode('audit',2,2,123)
    torch.testing.assert_close(inputs,again)


def test_packet_refuses_duplicate_generated_identities_and_capacity_overrun(tmp_path):
    args=packet_inputs(tmp_path)
    args.generated.append(args.generated[0])
    with pytest.raises(ValueError,match='Duplicate generated'):
        create_packet(args)
    assert not (tmp_path/'packet').exists()
    args.generated.pop()
    args.max_packet_gb=1e-6
    with pytest.raises(ValueError,match='exceeds cap'):
        create_packet(args)
    assert not (tmp_path/'packet').exists()


def test_synthetic_episode_cache_is_explicit_complete_and_non_overwriting(tmp_path):
    root=tmp_path/'synthetic'
    args=Namespace(output=str(root),synthetic=True,packet=None,source_repo=None,
        teacher_root=None,teacher_name='mae_latent_640',device='cpu',microbatch=2,
        queries=2,references=2,positions=2,train_episodes=2,val_episodes=1,
        audit_episodes=1,synthetic_width=2,cache_dtype='float32',max_cache_gb=.1,
        seed=42,cpu_threads=2)
    index=build_cache(args)
    assert index['complete'] and index['synthetic'] and len(index['episodes'])==4
    all_ids=[]
    noises={split:set() for split in ('train','val','audit')}
    positions={split:{stage:set() for stage in STAGES} for split in noises}
    for row in index['episodes']:
        item=torch.load(root/row['file'],weights_only=False,map_location='cpu')
        assert item['inputs'].shape==(4,4,32,32)
        assert len(item['teacher_features'])==15
        assert item['queries']==2
        for stage in STAGES:
            assert item['teacher_features'][stage+'/spatial'].shape==(2,4,index['teacher_channels'][stage])
            assert item['teacher_features'][stage+'/mean'].shape==(1,4,index['teacher_channels'][stage])
            assert torch.isfinite(item['teacher_features'][stage+'/std']).all()
        all_ids.extend(r['source_id'] for r in item['metadata']['source_records'])
        for r in item['metadata']['source_records']:
            if r['kind']=='generated':
                noises[row['split']].add(r['noise_id'])
                assert (r['checkpoint_id']=='synthetic_heldout')==(row['split']=='audit')
        for stage in STAGES:
            positions[row['split']][stage].update(item['positions'][stage].tolist())
    assert len(set(all_ids))==len(all_ids)
    assert noises['train'].isdisjoint(noises['val'])
    assert noises['train'].isdisjoint(noises['audit'])
    assert noises['val'].isdisjoint(noises['audit'])
    for stage in STAGES:
        assert positions['audit'][stage].isdisjoint(positions['train'][stage]|positions['val'][stage])
    assert index['actual_cache_bytes']==sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert index['teacher_construction_and_shape_check_seconds']>0
    assert index['cache_wall_seconds']>0
    with pytest.raises(FileExistsError):
        build_cache(args)


def test_packet_checksum_detects_changed_records(tmp_path):
    args=packet_inputs(tmp_path)
    create_packet(args)
    path=tmp_path/'packet'/'records.json'
    rows=json.loads(path.read_text())
    rows[0]['split']='audit' if rows[0]['split']!='audit' else 'train'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Packet identity changed'):
        Packet(args.output)
