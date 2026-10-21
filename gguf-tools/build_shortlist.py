#!/usr/bin/env python3
"""Build a drafter vocabulary from a declared, calibration-only UTF-8 corpus."""
import argparse,collections,json,os,subprocess
from pathlib import Path
from model_io import GGUF,sha256
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--corpus',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,default=32768);a=p.parse_args()
    if not 1<=a.size<=129280:p.error('invalid shortlist size')
    if a.out.exists():p.error('output exists')
    model=GGUF(a.model);counts=collections.Counter();env={k:v for k,v in os.environ.items() if not k.startswith(('LLMS_','DS4_'))};env['DS4_PROFILE']='upstream';env['LLMS_DS4_NO_ADOPTED']='1'
    for corpus in a.corpus:
        r=subprocess.run([str(ROOT/'ds4'),'-m',str(a.model.resolve()),'--dump-tokens','--raw','--prompt-file',str(corpus.resolve())],cwd=ROOT,env=env,capture_output=True,text=True,check=True,timeout=600)
        first=r.stdout.splitlines()[0];ids=json.loads(first)
        if not isinstance(ids,list) or not ids or any(not isinstance(x,int) or not 0<=x<129280 for x in ids):raise ValueError('tokenizer returned invalid IDs')
        counts.update(ids)
    ranked=sorted(counts,key=lambda x:(-counts[x],x));ids=sorted(set(ranked[:a.size])|set(range(128000,129280)))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(map(str,ids))+'\n')
    a.out.with_name(a.out.name+'.json').write_text(json.dumps({'model':model.identity(),'corpus_sha256':[sha256(f) for f in a.corpus],'size':len(ids)},indent=2)+'\n')
    if len(ranked)<a.size:print('WARNING: corpus has fewer distinct tokens than the requested shortlist; broaden the calibration corpus before measuring acceptance.')
if __name__=='__main__':main()
