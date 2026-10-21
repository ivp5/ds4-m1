#!/usr/bin/env python3
"""Build BS64 input importance from calibration-only FFN input captures."""
import argparse,json
from pathlib import Path
import numpy as np
from captures import records,E,NL
from model_io import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--captures',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    acc=np.zeros((NL,E),np.float64);count=np.zeros(NL,np.int64)
    for file in a.captures:
        for step,L,x,ids in records(file):acc[L]+=(x.astype(np.float64)**2).sum(0);count[L]+=len(x)
    if (count<64).any():p.error('need at least 64 captured rows at every layer; generate more calibration tokens')
    if a.out.exists():p.error('output exists')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('xb') as f:np.save(f,acc/count[:,None],allow_pickle=False)
    a.out.with_name(a.out.name+'.json').write_text(json.dumps({'rows_per_layer':count.tolist(),'capture_sha256':[sha256(x) for x in a.captures]},indent=2)+'\n')
if __name__=='__main__':main()
