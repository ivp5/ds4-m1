#!/usr/bin/env python3
"""Fit router lookahead residuals on independent calibration trajectories."""
import argparse,struct,os
from pathlib import Path
import numpy as np
import scipy.linalg as sl
from captures import records,E,NE,NL
from model_io import GGUF,receipt,sha256

def fit(X,XT,W,lam):
    G=X.T@X;G[np.diag_indices_from(G)]+=lam
    return W+sl.cho_solve(sl.cho_factor(G,lower=False,check_finite=False),X.T@(XT@W-X@W),check_finite=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--captures',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ridge',type=float,default=300.0);a=p.parse_args()
    if not np.isfinite(a.ridge) or a.ridge<=0:p.error('ridge must be positive')
    model=GGUF(a.model);sessions=[]
    for path in a.captures:
        by=[{} for _ in range(NL)]
        for step,L,x,ids in records(path):
            for row in range(len(x)):
                key=(step,row)
                if key in by[L]:raise ValueError('duplicate capture step/row; concatenate sessions as separate files')
                by[L][key]=x[row]
        sessions.append(by)
    a.out=a.out.resolve();a.out.parent.mkdir(parents=True,exist_ok=True);part=a.out.with_name(a.out.name+'.part')
    if a.out.exists() or part.exists():p.error('output exists')
    offsets=np.zeros((2,64),dtype='<u8');coverage=[]
    with a.model.open('rb') as src,part.open('xb') as f:
        f.write(b'\0'*16384)
        for h in (1,2):
            for L in range(NL-h):
                xx=[];tt=[]
                for by in sessions:
                    keys=sorted(by[L].keys() & by[L+h].keys())
                    if keys:xx.extend(by[L][k] for k in keys);tt.extend(by[L+h][k] for k in keys)
                if len(xx)<64:raise ValueError(f'layer {L}, horizon {h}: fewer than 64 aligned rows')
                X=np.array(xx,np.float64);XT=np.array(tt,np.float64)
                t=model.tensor(f'blk.{L+h}.ffn_gate_inp.weight',(E,NE),0);src.seek(t['offset']);b=src.read(E*NE*4)
                if len(b)!=E*NE*4:raise ValueError('short router read')
                W=np.frombuffer(b,'<f4').reshape(NE,E).T.astype(np.float64)
                A=np.ascontiguousarray(fit(X,XT,W,a.ridge).T.astype(np.float32))
                if not np.isfinite(A).all():raise ValueError('nonfinite fit')
                mx=np.abs(A).max(1);scale=np.where(mx>0,mx/np.float32(127),np.float32(1)).astype('<f4')
                q=np.clip(np.rint(A/scale[:,None]),-127,127).astype(np.int8)
                offsets[h-1,L+h]=f.tell();blob=q.tobytes()+scale.tobytes();f.write(blob);f.write(b'\0'*((-len(blob))%16384))
                coverage.append([L,h,len(X)]);print(f'layer {L} horizon {h}, {len(X)} aligned training rows',flush=True)
        hdr=bytearray(16384);struct.pack_into('<6I',hdr,0,0x46414c45,2,NL,NE,E,2);hdr[24:24+offsets.nbytes]=offsets.tobytes();f.seek(0);f.write(hdr);f.flush();os.fsync(f.fileno())
    part.rename(a.out);receipt(a.out,'ELAF-v2',model,ridge=a.ridge,capture_sha256=[sha256(x) for x in a.captures],coverage=coverage)
if __name__=='__main__':main()
