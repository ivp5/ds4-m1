#!/usr/bin/env python3
"""Stream official FP4 gate/up tensors into the production BS64 layout."""
import argparse, json, os, shutil, struct, time
from pathlib import Path
import numpy as np
import bs64_codec as codec
from model_io import GGUF, SafeTensors, receipt, sha256

def load(source,layer,expert,which):
    name=f'layers.{layer}.ffn.experts.{expert}.{which}'
    raw,wt=source.raw(name+'.weight');scale,st=source.raw(name+'.scale')
    if len(raw)!=codec.ROWS*codec.COLS//2 or len(scale)!=codec.ROWS*codec.COLS//32:raise ValueError('unexpected FP4 tensor layout')
    a=np.frombuffer(raw,np.uint8);q=np.stack([a&15,a>>4],axis=1).reshape(codec.ROWS,codec.COLS)
    lut=np.array([0,.5,1,1.5,2,3,4,6],np.float32)
    w=lut[q&7]*np.where(q&8,np.float32(-1),np.float32(1))
    sc=np.frombuffer(scale,np.uint8).reshape(codec.ROWS,codec.COLS//32)
    w*=np.repeat(np.exp2(sc.astype(np.float32)-127),32,axis=1)
    if not np.isfinite(w).all():raise ValueError('nonfinite FP4 input')
    return w

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('hf','model','importance','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--revision',required=True,help='Exact source checkpoint revision')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--plan',action='store_true');a=p.parse_args()
    model=GGUF(a.model);source=SafeTensors(a.hf)
    imp=np.load(a.importance,allow_pickle=False).astype(np.float64)
    if imp.shape!=(40,5120) or not np.isfinite(imp).all() or (imp<0).any() or not (imp.sum(1)>0).all():p.error('importance must be finite, nonnegative float [40,5120] with nonzero rows')
    for L in range(40):
        model.tensor(f'blk.{L}.ffn_gate_exps.weight',(5120,2304,384))
        for e in range(384):
            for w in ('w1','w3'):
                for suffix in ('weight','scale'):
                    name=f'layers.{L}.ffn.experts.{e}.{w}.{suffix}'
                    if name not in source.mapping:p.error('missing source tensor '+name)
    size=codec.HDR+40*codec.LAYB;print(f'Output {size:,} bytes; source revision {a.revision}',flush=True)
    if a.plan:return
    a.out=a.out.resolve();a.out.parent.mkdir(parents=True,exist_ok=True)
    part=a.out.with_name(a.out.name+'.part')
    if a.out.exists() or part.exists():p.error('output or partial output already exists; inspect before retrying')
    if shutil.disk_usage(a.out.parent).free<size+(1<<30):p.error('insufficient free disk space')
    encode=codec.encode_np
    if a.device=='cuda':
        import torch
        codec.torch=torch;codec.dev=torch.device('cuda');codec.T64=torch.float64
        def accelerated(w,importance):
            b,ew,ep,bad=codec.encode_t(torch.tensor(w,device=codec.dev),torch.tensor(importance,device=codec.dev,dtype=torch.float64))
            if bad:raise ValueError('nonfinite encoded values')
            return b
        w=load(source,0,0,'w1');ref=codec.encode_np(w,imp[0]);got=accelerated(w,imp[0])
        if got!=ref:p.error('GPU codec differs from the NumPy reference; rerun with --device cpu')
        encode=accelerated;print('GPU/reference byte equality: PASS',flush=True)
    header=struct.pack('<4sIIIIIQQQQ',b'BS64',1,40,384,2304,5120,codec.MATB,codec.EXPB,codec.LAYB,codec.HDR)
    t=time.monotonic()
    with part.open('xb') as f:
        f.write(header+b'\0'*(codec.HDR-len(header)))
        # On-disk order: all gates, then all ups, for each layer.
        for L in range(40):
            for which in ('w1','w3'):
                for e in range(384):
                    b=encode(load(source,L,e,which),imp[L])
                    if len(b)!=codec.MATB:raise ValueError('wrong encoded matrix length')
                    f.write(b)
            f.flush();os.fsync(f.fileno());print(f'layer {L+1}/40; {time.monotonic()-t:.0f} s',flush=True)
    if part.stat().st_size!=size:raise ValueError('wrong final BS64 length')
    part.rename(a.out)
    receipt(a.out,'BS64-v1',model,checkpoint_revision=a.revision,importance_sha256=sha256(a.importance),source_index_sha256=sha256(a.hf/'model.safetensors.index.json'),implementation=a.device)
if __name__=='__main__':main()
