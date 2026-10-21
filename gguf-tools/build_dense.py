#!/usr/bin/env python3
"""Stream Q8_0 dense / F16 Engram weights directly to a Q4L side file."""
import argparse, json, math, os, re, struct
from pathlib import Path
import numpy as np
from dense_codec import quant
from model_io import GGUF, receipt, sha256
CLASSES={'attn_q_b','attn_output_b','ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp','output','attn_output_a','attn_q_a','attn_kv'}
def load_stats(path,model=None):
    path=Path(path)
    if path.suffix=='.npz':
        if model is None:raise ValueError('named statistics require --model')
        meta=json.loads(path.with_name('manifest.json').read_text())
        if meta.get('format')!='ds4-activation-mean-v1' or sha256(path)!=meta['files'][path.name]['sha256']:
            raise ValueError('calibration manifest/hash mismatch')
        out={}
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!=set(meta['tensor_shapes']):raise ValueError('calibration tensor coverage mismatch')
            for name in data.files:
                t=model.tensor(name,meta['tensor_shapes'][name]);x=data[name]
                if x.ndim!=1 or not len(x) or len(x)%t['shape'][0] or not np.isfinite(x).all() or (x<0).any():
                    raise ValueError('invalid named activation statistics')
                out[t['offset']]=x.astype(np.float32)
        return out
    side=path.with_name(path.name+'.json')
    if model is not None and side.exists() and json.loads(side.read_text()).get('model')!=model.identity():
        raise ValueError('activation statistics belong to another GGUF; collect again or use named statistics')
    b=Path(path).read_bytes()
    if b[:4]!=b'ACTS' or len(b)<8:raise ValueError('expected ACTS activation statistics')
    n=struct.unpack_from('<I',b,4)[0];pos=8;out={}
    for _ in range(n):
        off,k,count=struct.unpack_from('<QII',b,pos);pos+=16
        x=np.frombuffer(b,'<f4',k,pos).copy();pos+=4*k
        if not count or off in out or not np.isfinite(x).all() or (x<0).any():raise ValueError('bad ACTS record')
        out[off]=x/count
    if pos!=len(b) or not out:raise ValueError('empty/trailing ACTS records')
    return out

def pack(q,d,K,N):
    q=(q.astype(np.int16)+8).astype(np.uint8).reshape(N,K//32,32)
    nib=(q[:,:,0::2]|(q[:,:,1::2]<<4)).reshape(N,K//2)
    scales=np.asarray(d,dtype='<f2').reshape(N,K//32).view(np.uint8)
    result=np.concatenate([nib,scales],axis=1)
    if result.shape!=(N,9*K//16):raise ValueError('packing shape')
    return result

def quant_eng(w):
    w=w.reshape(-1,32);amax=np.abs(w).max(1,keepdims=True);base=np.where(amax>0,amax/7.5,1.0)
    best=np.full(amax.shape,np.inf,np.float32);scale=base.copy()
    for c in np.linspace(.85,1.25,24):
        s=base*c;q=np.clip(np.rint(w/s),-8,7);err=((w-q*s)**2).sum(1,keepdims=True);take=err<best
        best=np.where(take,err,best);scale=np.where(take,s,scale)
    d=scale.astype(np.float16);q=np.clip(np.rint(w/np.where(d.astype(np.float32)>0,d.astype(np.float32),1.0)),-8,7).astype(np.int8)
    return d.reshape(-1),q

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stats',type=Path);p.add_argument('--draft',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    model=GGUF(a.model);stats=load_stats(a.stats,model) if a.stats else {}
    if not a.draft and not stats:p.error('--stats is required for main-model importance weighting; collect it on a separate calibration corpus')
    tensors=[]
    for name,t in model.tensors.items():
        cls=re.sub(r'^blk\.\d+\.','',name).removesuffix('.weight');shape=t['shape']
        eligible=t['type']==8 and len(shape)==2 and shape[0]%256==0 if a.draft else cls in CLASSES or cls=='engram_kv'
        if not eligible:continue
        if len(shape)!=2 or shape[0]%256 or t['type'] not in (8,1):raise ValueError(f'{name}: incompatible Q4L layout')
        tag='q4le' if cls=='engram_kv' else 'q4la' if cls=='attn_output_a' else 'q4lq' if cls in ('attn_q_a','attn_kv') else ''
        tensors.append((name,t,tag))
    if not tensors:p.error('no compatible tensors')
    size=0
    for _,t,_ in tensors:size=(size+16383)//16384*16384+9*math.prod(t['shape'])//16
    print(f'{len(tensors)} tensors; {size:,} output bytes',flush=True)
    if a.plan:return
    a.out=a.out.resolve();a.out.parent.mkdir(parents=True,exist_ok=True);idx=a.out.with_suffix('.idx');part=a.out.with_name(a.out.name+'.part')
    if any(x.exists() for x in (a.out,part,idx)):p.error('output/index exists; choose an unused destination')
    import shutil
    if shutil.disk_usage(a.out.parent).free<size+(1<<28):p.error('insufficient free space')
    rows=[];uncalibrated=[]
    with a.model.open('rb') as src,part.open('xb') as dest:
        for name,t,tag in tensors:
            K,N=t['shape'];dest.write(b'\0'*((-dest.tell())%16384));start=dest.tell();act=stats.get(t['offset'])
            if act is not None and (len(act)%K or not len(act)):raise ValueError('activation group shape mismatch')
            if not a.draft and t['type']==8 and act is None:uncalibrated.append(name)
            for r0 in range(0,N,128):
                nr=min(128,N-r0);rb=K//32*34 if t['type']==8 else K*2;src.seek(t['offset']+r0*rb);raw=src.read(nr*rb)
                if len(raw)!=nr*rb:raise ValueError('short tensor read')
                if t['type']==8:
                    arr=np.frombuffer(raw,np.uint8).reshape(-1,34)
                    w=arr[:,2:].copy().view(np.int8).astype(np.float32)*arr[:,:2].copy().view('<f2').astype(np.float32)
                    if a.draft:
                        ix=np.abs(w).argmax(1);mx=w[np.arange(len(w)),ix];d=(mx/-8).astype(np.float16);df=d.astype(np.float32)
                        inv=np.where(df!=0,1/np.where(df!=0,df,1),0);q=(np.clip(np.round(w*inv[:,None])+8,0,15)-8).astype(np.int8)
                    else:
                        importance=None
                        if act is not None:
                            bi=np.arange(r0*(K//32),(r0+nr)*(K//32));rr=bi//(K//32);grp=rr*(len(act)//K)//N;col=(bi%(K//32))*32
                            importance=act[(grp*K+col)[:,None]+np.arange(32)[None,:]]
                        d,q=quant(w,importance)
                else:
                    w=np.frombuffer(raw,'<f2').astype(np.float32).reshape(-1,32);d,q=quant_eng(w)
                if not np.isfinite(d).all():raise ValueError('nonfinite scale')
                b=pack(q,d,K,nr);dest.write(b.tobytes())
            rows.append(f'{t["offset"]} {start} {K} {N} {name}'+(' '+tag if tag and not a.draft else ''))
            print(name,flush=True)
        dest.flush();os.fsync(dest.fileno())
    if part.stat().st_size!=size:raise ValueError('unexpected output length')
    part.rename(a.out);idx.write_text(a.out.name+f'\ngguf_bytes {model.bytes}\n'+'\n'.join(rows)+'\n')
    receipt(a.out,'Q4L-row-v1',model,stats_sha256=sha256(a.stats) if a.stats else None,uncalibrated_tensors=uncalibrated,draft=a.draft)
    if uncalibrated:print(f'WARNING: {len(uncalibrated)} tensors had no activation statistics; uniformly weighted. Validate fidelity independently.')
if __name__=='__main__':main()
