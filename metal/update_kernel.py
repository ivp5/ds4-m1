#!/usr/bin/env python3
"""Record a complete reviewed Metal variant as a small patch against its base."""
import argparse,difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def entry(name,base_name,base_bytes,variant):
    if Path(name).name!=name or not name.endswith('.metal'):raise ValueError('variant must be a .metal basename')
    if not base_name.startswith(('metal/','build/metal/')) or '..' in Path(base_name).parts:raise ValueError('base must be under metal/ or a generated kernel under build/metal/')
    patch=''.join(difflib.unified_diff(base_bytes.decode().splitlines(keepends=True),variant.decode().splitlines(keepends=True),
                                    fromfile='a/'+name,tofile='b/'+name,n=3))
    if not patch:raise ValueError('variant equals base; use the base source directly instead')
    h=lambda b:hashlib.sha256(b).hexdigest()
    return patch,dict(base=base_name,base_sha256=h(base_bytes),patch=name+'.patch',patch_sha256=h(patch.encode()),
                      output_sha256=h(variant),output_bytes=len(variant))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--base',help='Required for a new variant: canonical metal/ source or generated build/metal/ parent');a=p.parse_args()
    dest=ROOT/'metal/patches';manifest=json.loads((dest/'manifest.json').read_text());old=manifest.get(a.name,{})
    base=a.base or old.get('base')
    if not base:p.error('--base is required for a new variant')
    try:patch,e=entry(a.name,base,(ROOT/base).read_bytes(),a.source.read_bytes())
    except (OSError,ValueError) as ex:p.error(str(ex))
    (dest/e['patch']).write_text(patch);manifest[a.name]=e
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Recorded',a.name,'with output SHA-256',e['output_sha256'])
    print('Run make metal-variants and the checks in docs/MAINTAINING.md before committing.')
if __name__=='__main__':main()
