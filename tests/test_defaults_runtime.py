"""Compare the compiled constructor with all three canonical profiles.
Links only the help/defaults object: no engine constructor, GPU or model is run.
"""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gguf-tools'))
from generate_defaults import profiles as load_profiles
def main():
    out=ROOT/'build/checks/profile-test';out.mkdir(parents=True,exist_ok=True)
    tmp=out/'tmp';tmp.mkdir(exist_ok=True)
    c=out/'dump.c';c.write_text('#include <stdio.h>\n#include <string.h>\nextern char **environ;\n/* Defaults-only test double: this executable contains no inference engine. */\nint llms_mg_holds_engine_lock(void){return 1;}\nint llms_profile_check_assets(void);\nint main(int argc,char **argv){if(argc>1&&!strcmp(argv[1],"--check-assets"))return llms_profile_check_assets()?0:78;for(char **e=environ;*e;e++)if(!strncmp(*e,"LLMS_",5)||!strncmp(*e,"DS4_METAL_",10))puts(*e);return 0;}\n')
    exe=out/'dump';env=dict(os.environ,TMPDIR=str(tmp))
    subprocess.run(['cc','-o',str(exe),str(c),str(ROOT/'ds4_help.o')],check=True,env=env)
    profiles=load_profiles()
    base={k:v for k,v in os.environ.items() if not k.startswith(('LLMS_','DS4_'))}
    base.update(DS4_PROFILE='metal64',DS4_ASSET_ROOT=str(ROOT),DS4_MODEL_DIR=str(out/'models with spaces'),DS4_DRAFT_DIR=str(out/'draft with spaces'))
    for kind,name in [('agent','ds4-agent'),('cli','ds4'),('server','ds4-server')]:
        link=out/name
        if not link.exists():link.symlink_to(exe.name)
        for mode,override in [('metal64',False),('metal64',True),('',False),(None,False)]:
            e=dict(base)
            if mode is None:e.pop('DS4_PROFILE')
            else:e['DS4_PROFILE']=mode
            if override:e['LLMS_DS4_CD_ALL']='987'
            r=subprocess.run([str(link)],cwd=out,env=e,text=True,capture_output=True,check=True)
            got=dict(line.split('=',1) for line in r.stdout.splitlines())
            expected={k:v.replace('@repo/',str(ROOT)+'/').replace('@model/',base['DS4_MODEL_DIR']+'/').replace('@draft/',base['DS4_DRAFT_DIR']+'/') for k,v in profiles[kind].items()}
            if override:expected['LLMS_DS4_CD_ALL']='987'
            if got!=expected:raise AssertionError((kind,{k:(expected.get(k),got.get(k)) for k in expected.keys()|got.keys() if expected.get(k)!=got.get(k)}))
        print(kind,'compiled defaults and explicit override: PASS')
    for mode in ('upstream',):
        e=dict(base);e['DS4_PROFILE']=mode
        r=subprocess.run([str(exe)],env=e,capture_output=True,text=True,check=True)
        if r.stdout.strip():raise AssertionError('upstream mode installed optimized defaults')
    print('upstream mode: PASS')
    missing = dict(base); missing.pop('DS4_PROFILE')
    # A unique empty directory proves this result cannot depend on a previous run's files.
    import tempfile
    with tempfile.TemporaryDirectory(dir=out) as empty:
        missing['DS4_MODEL_DIR']=empty; missing['DS4_DRAFT_DIR']=str(Path(empty)/'draft')
        r=subprocess.run([str(out/'ds4-agent'),'--check-assets'],env=missing,capture_output=True,text=True)
        if r.returncode != 78 or 'bs64_gateup.bin' not in r.stderr or 'docs/METAL_SETUP.md' not in r.stderr:
            raise AssertionError(('missing assets were not explained',r.returncode,r.stderr))
        for value in profiles['agent'].values():
            if value.startswith('@model/') or value.startswith('@draft/'):
                path=Path(value.replace('@model/',empty+'/').replace('@draft/',str(Path(empty)/'draft')+'/'))
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'presence-only fixture')
        r=subprocess.run([str(out/'ds4-agent'),'--check-assets'],env=missing,capture_output=True,text=True)
        if r.returncode:raise AssertionError(r.stderr)
        Path(empty,'g32_down.bin').write_bytes(b'')
        r=subprocess.run([str(out/'ds4-agent'),'--check-assets'],env=missing,capture_output=True,text=True)
        if r.returncode != 78 or 'g32_down.bin' not in r.stderr:raise AssertionError('empty asset accepted')
        missing['LLMS_DS4_DOWN_G32']='0'
        subprocess.run([str(out/'ds4-agent'),'--check-assets'],env=missing,check=True,capture_output=True)
    print('native default, missing/empty asset errors and explicit disabled asset: PASS')
if __name__=='__main__':main()
