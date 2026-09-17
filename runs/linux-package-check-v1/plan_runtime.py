"""Inspect small public metadata for a local CPU-only Linux reproduction."""
import hashlib,json,urllib.request,urllib.parse,urllib.error
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs))
def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'airtrafficcontrol-reproduction-check'})
    with urllib.request.urlopen(request,timeout=30) as response:return response.read()
index_url='https://download.pytorch.org/whl/cpu/torch/'
raw=fetch(index_url);(out/'torch-cpu-index.html').write_bytes(raw)
parser=Links();parser.feed(raw.decode())
filename='torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl'
matches=[a for a in parser.links if urllib.parse.unquote(urllib.parse.urlsplit(a['href']).path).endswith('/'+filename)]
assert len(matches)==1,matches
link=matches[0];url=urllib.parse.urljoin(index_url,link['href']);parts=urllib.parse.urlsplit(url)
digest=urllib.parse.parse_qs(parts.fragment)['sha256'][0]
wheel_url=urllib.parse.urlunsplit(parts._replace(fragment=''))
request=urllib.request.Request(wheel_url,method='HEAD')
errors={}
try:
    with urllib.request.urlopen(request,timeout=30) as response:headers=dict(response.headers)
except urllib.error.HTTPError as error:
    headers={};errors['wheel_head_http_status']=error.code
try:
    metadata=fetch(wheel_url+'.metadata');(out/'torch-cpu-wheel-metadata.txt').write_bytes(metadata)
    requirements=[line for line in metadata.decode().splitlines() if line.startswith('Requires-Dist:')]
    assert not any('nvidia-' in line.lower() or 'triton' in line.lower() for line in requirements)
except urllib.error.HTTPError as error:
    metadata=None;requirements=None;errors['wheel_metadata_http_status']=error.code
release_url='https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest'
release_raw=fetch(release_url);(out/'python-standalone-release.json').write_bytes(release_raw)
release=json.loads(release_raw)
assets=[{k:a.get(k) for k in ('name','size','browser_download_url','digest')} for a in release['assets']
        if a['name'].startswith('cpython-3.12.14+') and '-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz' in a['name']]
assert len(assets)<=1,assets
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
 'torch_cpu':{'filename':filename,'url':wheel_url,'sha256':digest,
              'compressed_bytes':int(headers.get('Content-Length',headers.get('content-length',0))),
              'metadata_sha256':hashlib.sha256(metadata).hexdigest() if metadata else None,'requires_dist':requirements,
              'gpu_dependencies_declared':False if requirements is not None else None,'http_errors':errors},
 'python_release':release['tag_name'],'matching_python_3_12_14_assets':assets,
 'binary_archives_downloaded':False,'environment_created':False,'linux_code_executed':False}
(out/'runtime-plan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
