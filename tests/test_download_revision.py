"""Offline download workflow test using a tiny fixture, never fetching model weights."""
import hashlib,json,os,re,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Download(unittest.TestCase):
    def test_revision_checksum_reuse_and_symlink(self):
        local=ROOT/'build/checks';local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as d:
            root=Path(d);bin_dir=root/'bin';bin_dir.mkdir()
            fixture=b'test GGUF payload\n';source=(ROOT/'download_model.sh').read_text()
            # Substitute only the Q2 artifact identity, keeping the production workflow.
            source,n=re.subn(r'(\"\$DS41_Q2_FILE\"\)\n\s*expected_bytes=)\d+(\n\s*expected_sha=)[a-f0-9]+',
                             lambda m:m[1]+str(len(fixture))+m[2]+hashlib.sha256(fixture).hexdigest(),source)
            self.assertEqual(n,1)
            script=root/'download_model.sh';script.write_text(source)
            hf=bin_dir/'hf';hf.write_text("#!/usr/bin/env python3\nimport sys,json\nfrom pathlib import Path\na=sys.argv[1:]\nout=Path(a[a.index('--local-dir')+1]);out.mkdir(parents=True,exist_ok=True)\n(out/'invocation.json').write_text(json.dumps(a))\n(out/a[2]).write_bytes("+repr(fixture)+")\n")
            hf.chmod(0o755);env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
            cmd=['sh',str(script),'ds41f-q2','--token','fixture-token']
            first=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(first.returncode,0,first.stderr)
            link=root/'ds4flash.gguf';self.assertTrue(link.is_symlink());self.assertEqual(link.read_bytes(),fixture)
            args=json.loads((root/'gguf/invocation.json').read_text())
            self.assertEqual(args[args.index('--revision')+1],'9dcee51692e6540f697141fc98b097e3c96492a3')
            # An existing verified file must not need a downloader at all.
            hf.unlink()
            second=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(second.returncode,0,second.stderr)
            link.write_bytes(b'X'*len(fixture))
            corrupt=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(corrupt.returncode,0);self.assertIn('Checksum mismatch',corrupt.stderr)
if __name__=='__main__':unittest.main()
