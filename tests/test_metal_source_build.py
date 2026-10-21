"""Compile the actual concatenated Metal libraries offline; no GPU model run."""
import ast,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gguf-tools'));sys.path.insert(0,str(ROOT/'metal'))
from generate_defaults import profiles as load_profiles
from build_kernels import build
build()
OUT=ROOT/'build/metal-check';OUT.mkdir(parents=True,exist_ok=True)
tmp=ROOT/'build/tmp';tmp.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,TMPDIR=str(tmp),CLANG_MODULE_CACHE_PATH=str(ROOT/'build/module-cache'))
s=(ROOT/'ds4_metal.m').read_text();i=s.index('static const char *ds4_gpu_source =');j=s.index(';\n',i)
base=''.join(ast.literal_eval('"'+v+'"') for v in re.findall(r'"((?:[^"\\]|\\.)*)"',s[i:j]))
k=s.index('NSArray<NSArray<NSString *> *> *required_sources = @[')
end=s.index('];',k)
specs=re.findall(r'@\[@"([A-Z0-9_]+)",\s*@"(metal/[^"]+)"\]',s[k:end])
if len(specs)<10:raise SystemExit('Could not parse the complete Metal source list')
profiles=load_profiles();seen=set();results=[]
for name,profile in [('upstream',{}),*profiles.items()]:
    files=[]
    for key,path in specs:
        value=profile.get(key,path)
        if value.startswith('@repo/'):value=value[6:]
        files.append(value)
    identity=tuple(files)
    if identity in seen:print(name,': identical source assembly already checked');continue
    seen.add(identity);text=base+''.join('\n'+(ROOT/f).read_text()+'\n' for f in files)
    src=OUT/(name+'.metal');src.write_text(text)
    cmd=['xcrun','-sdk','macosx','metal','-c','-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1',str(src),'-o',str(OUT/(name+'.air'))]
    run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
    (OUT/(name+'.log')).write_text(run.stdout+run.stderr)
    print(name,len(files),'sources, compiler return',run.returncode,flush=True)
    results.append(dict(profile=name,returncode=run.returncode,sources=files))
    if run.returncode:print(run.stderr[-4000:]);raise SystemExit(1)
# Negative control must fail using the same compiler invocation.
src.write_text(text+'\nthis line cannot compile;\n')
run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
print('syntax corruption canary:', 'PASS' if run.returncode else 'FAIL',flush=True)
(OUT/'results.json').write_text(json.dumps(dict(results=results,canary_failed=bool(run.returncode)),indent=2)+'\n')
if not run.returncode:raise SystemExit(1)
