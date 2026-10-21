#!/usr/bin/env python3
"""Collect calibration artifacts on explicitly supplied, non-evaluation prompts.
Never use evaluation tasks, their outputs, or private logs as calibration input.
"""
import argparse,json,os,struct,subprocess,time
from pathlib import Path
import numpy as np
from model_io import GGUF,sha256
ROOT=Path(__file__).resolve().parents[1]

def bootstrap_env():
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLMS_','DS4_'))}
    env.update(DS4_PROFILE='metal64',DS4_ASSET_ROOT=str(ROOT))
    env.update({k:'0' for k in ['LLMS_DS4_GATEUP_BS64','LLMS_DS4_DECODE_EXPERTS_BS64','LLMS_DS4_DOWN_G32','LLMS_DS4_Q4L','LLMS_DS4_Q4L_DSP','LLMS_DS4_DSP41','LLMS_DS4_ELA_FIT','LLMS_DS4_SHM_CACHE','LLMS_DS4_DSP_BANK_SHM','LLMS_DS4_Q4G','LLMS_DS4_CED_SP']})
    env.update(LLMS_DS4_LAG='0',LLMS_DS4_DSP41_BANK='',LLMS_DS4_DSP41_SHORT='',LLMS_DS4_EXPERT_CACHE_SLOT_BYTES='10387456',
               LLMS_DS4_EXPERT_CACHE_GIB='24',LLMS_DS4_EXPERT_CACHE_EXPERIMENT='1',
               LLMS_DS4_ROUTING='exact',LLMS_DS4_DECODE_ROUTE='0',LLMS_DS4_DECODE_ROUTE_LATE='0',LLMS_DS4_DECODE_ROUTE_TOPK='0',
               LLMS_DS4_PREFILL_ROUTE='0',LLMS_DS4_PREFILL_ROUTE_SHORT='0')
    return env

def merge_stats(files,out):
    acc={}
    for path in files:
        b=path.read_bytes()
        if b[:4]!=b'ACTS':raise ValueError('ACTS collector did not run')
        n=struct.unpack_from('<I',b,4)[0];pos=8
        for _ in range(n):
            off,k,c=struct.unpack_from('<QII',b,pos);pos+=16;x=np.frombuffer(b,'<f4',k,pos).astype(np.float64);pos+=4*k
            if not c or not np.isfinite(x).all():raise ValueError('invalid ACTS sample')
            if off not in acc:acc[off]=[0,np.zeros(k,np.float64)]
            if len(acc[off][1])!=k:raise ValueError('incompatible ACTS shapes')
            acc[off][0]+=c;acc[off][1]+=x
        if pos!=len(b):raise ValueError('trailing ACTS data')
    if not acc:raise ValueError('no Q8 dense activations captured; inspect per-prompt log')
    with out.open('xb') as f:
        f.write(b'ACTS'+struct.pack('<I',len(acc)))
        for off,(c,x) in sorted(acc.items()):f.write(struct.pack('<QII',off,len(x),c));f.write(x.astype('<f4').tobytes())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--prompts',type=Path,default=ROOT/'gguf-tools/calibration_prompts.json');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=256);p.add_argument('--draft',type=Path);p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    prompts=json.loads(a.prompts.read_text())
    if not isinstance(prompts,list) or not prompts or any(not isinstance(x,str) or not x.strip() for x in prompts):p.error('prompts must be a nonempty JSON array of strings')
    if a.out.exists():p.error('output directory exists; use a new directory to avoid mixing sessions')
    a.out=a.out.resolve();a.out.mkdir(parents=True)
    model=GGUF(a.model);env=bootstrap_env();statfiles=[];runs=[]
    if a.draft:
        env['LLMS_DS4_DSP41']=str(a.draft.resolve())
        # An unrestricted drafter gives unbiased counts for bank selection.
        env['LLMS_DS4_DSP41_BANK']='';env['LLMS_DS4_DSP41_SHORT']=''
    for i,prompt in enumerate(prompts):
        base=a.out/f'{i:03d}';pf=base.with_suffix('.prompt.txt');pf.write_text(prompt)
        stats=base.with_suffix('.acts');capture=base.with_suffix('.xcap');counts=base.with_suffix('.counts')
        e=dict(env,LLMS_DS4_ACT_STATS=str(stats),LLMS_DS4_ELA_XCAP=str(capture))
        if a.draft:e['LLMS_DS4_DSP41_COUNT']=str(counts)
        cmd=[str(ROOT/'ds4'),'-m',str(a.model.resolve()),'--prompt-file',str(pf),'-n',str(a.tokens),'-c','8192','--temp','0','--seed','12345']
        started=time.monotonic()
        with base.with_suffix('.stdout').open('w') as out,base.with_suffix('.stderr').open('w') as err:
            r=subprocess.run(cmd,cwd=ROOT,env=e,stdout=out,stderr=err,timeout=a.timeout)
        runs.append({'prompt_sha256':sha256(pf),'returncode':r.returncode,'seconds':time.monotonic()-started})
        (a.out/'runs.json').write_text(json.dumps({'model':model.identity(),'prompt_source_sha256':sha256(a.prompts),'runs':runs},indent=2)+'\n')
        if r.returncode:raise RuntimeError(f'calibration session {i} failed; inspect {base.name}.stderr')
        if not capture.is_file() or capture.stat().st_size==0:raise RuntimeError('FFN capture did not execute; inspect stderr rather than training on an empty file')
        if a.draft and (not counts.is_file() or not counts.stat().st_size):raise RuntimeError('drafter counts missing')
        statfiles.append(stats);print(f'prompt {i+1}/{len(prompts)} complete',flush=True)
    merge_stats(statfiles,a.out/'dense.acts')
    (a.out/'dense.acts.json').write_text(json.dumps({'model':model.identity(),'calibration_sha256':sha256(a.prompts)},indent=2)+'\n')
if __name__=='__main__':main()
