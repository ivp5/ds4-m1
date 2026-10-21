"""Preflight for the single-user 64-GiB Metal profile. No GPU work."""
import json,struct
from pathlib import Path
from generate_defaults import profiles
from model_io import GGUF
ROOT=Path(__file__).resolve().parents[1]

def indexed(idx,model,errors):
    try:
        lines=idx.read_text().splitlines()
        if len(lines)<3 or lines[1]!=f'gguf_bytes {model.bytes}':raise ValueError('index/model size mismatch or empty index')
        payload=Path(lines[0]);payload=payload if payload.is_absolute() else idx.parent/payload
        size=payload.stat().st_size;seen=set()
        for line in lines[2:]:
            a,b,k,n,name,*tag=line.split();a,b,k,n=map(int,(a,b,k,n));t=model.tensor(name)
            if a!=t['offset'] or tuple(t['shape'])!=(k,n) or k%256 or b%16384 or b<0 or b+9*k*n//16>size or name in seen:raise ValueError('wrong tensor offsets/shapes/ranges')
            seen.add(name)
        return payload
    except (OSError,ValueError,KeyError,IndexError) as e:errors.append(f'{idx.name}: {e}');return None

def validate(model_path,model_dir,draft_dir,kind='cli',require_receipts=True):
    model=GGUF(model_path);md=Path(model_dir);dd=Path(draft_dir);errors=[];assets={}
    profile=profiles()[kind]
    for k,v in profile.items():
        for prefix,base in [('@repo/',ROOT),('@model/',md),('@draft/',dd)]:
            if v.startswith(prefix):
                path=base/v[len(prefix):];assets[k]=path
                if not path.is_file():errors.append(f'missing {path.name}')
    if errors:return errors,assets
    try:
        for L in range(40):
            model.tensor(f'blk.{L}.ffn_down_exps.weight',(2304,5120,384),10)
            model.tensor(f'blk.{L}.ffn_gate_exps.weight',(5120,2304,384),16)
        bs=assets['LLMS_DS4_GATEUP_BS64']
        with bs.open('rb') as f:h=struct.unpack('<4sIIIIIQQQQ',f.read(56))
        if h!=(b'BS64',1,40,384,2304,5120,3225600,6451200,2477260800,16384) or bs.stat().st_size!=99090448384:raise ValueError('BS64 header/size mismatch')
        generated=[bs]
        if kind!='server':
            g32=assets['LLMS_DS4_DOWN_G32']
            with g32.open('rb') as f:h=struct.unpack('<4sIIIIIQQQQQQ',f.read(72))
            if h!=(b'G32D',1,40,384,5120,2304,3502080,1344798720,16384,0,2949120,3317760) or g32.stat().st_size!=53791965184:raise ValueError('G32 header/size mismatch')
            dense=indexed(assets['LLMS_DS4_Q4L'],model,errors)
            support=GGUF(assets['LLMS_DS4_DSP41']);draft=indexed(assets['LLMS_DS4_Q4L_DSP'],support,errors)
            for st in range(3):support.tensor(f'mtp.{st}.ffn_gate_exps.weight',(5120,2304,128),16)
            bank=[[int(i) for i in l.split()] for l in assets['LLMS_DS4_DSP41_BANK'].read_text().splitlines()]
            if len(bank)!=3 or any(len(r)<3 or len(r)!=len(set(r)) or min(r)<0 or max(r)>=128 for r in bank):raise ValueError('bank must contain three unique expert-ID rows, at least three per stage')
            ids=[int(i) for i in assets['LLMS_DS4_DSP41_SHORT'].read_text().split()]
            if not ids or len(ids)!=len(set(ids)) or min(ids)<0 or max(ids)>=129280:raise ValueError('invalid drafter shortlist')
            ela=assets['LLMS_DS4_ELA_FIT']
            with ela.open('rb') as f:b=f.read(16384)
            if struct.unpack_from('<6I',b)!=(0x46414c45,2,40,384,5120,2):raise ValueError('lookahead header mismatch')
            off=struct.unpack_from('<128Q',b,24);table=384*5120+384*4
            if sum(bool(x) for x in off)!=77 or any(x and (x%16384 or x+table>ela.stat().st_size) for x in off):raise ValueError('lookahead table coverage/range mismatch')
            generated += [g32,ela]+([dense] if dense else [])
            if draft:check_receipt(draft,support,errors,require_receipts)
        for path in generated:check_receipt(path,model,errors,require_receipts)
    except (OSError,ValueError,KeyError,struct.error) as e:errors.append(str(e))
    return errors,assets

def check_receipt(path,model,errors,required):
    file=path.with_name(path.name+'.json')
    if not file.exists():
        if required:errors.append(f'missing {file.name}; regenerate the asset using the README recipe')
        return
    try:
        data=json.loads(file.read_text())
        if data.get('model')!=model.identity() or data.get('bytes')!=path.stat().st_size:raise ValueError('asset/model metadata mismatch')
    except (OSError,ValueError) as e:errors.append(f'{file.name}: {e}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT/'ds4flash.gguf')
    p.add_argument('--assets', type=Path, default=ROOT/'gguf/ds41')
    p.add_argument('--program', choices=['cli','agent','server'], default='agent')
    a = p.parse_args()
    try: errors, _ = validate(a.model, a.assets, a.assets/'draft', a.program)
    except (OSError, ValueError) as e: errors = [str(e)]
    if errors: p.exit(2, '\n'.join(errors)+'\nSee docs/METAL_SETUP.md.\n')
    print('Optimized assets: ready. Run ./ds4-agent (or ./ds4 / ./ds4-server).')
