"""Inspect published Linux packages without installing or executing them."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,importlib.metadata as md,json,urllib.request,urllib.error,zipfile
from packaging.tags import cpython_tags,compatible_tags
from packaging.utils import parse_wheel_filename
root=Path.cwd()
out=root/'runs/linux-package-check-v1'
platforms=[f'manylinux_2_{minor}_x86_64' for minor in range(35,4,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
tags=set(cpython_tags((3,12),abis=['cp312'],platforms=platforms))|set(compatible_tags((3,12),interpreter='cp312',platforms=platforms))
names=['bluesky-simulator','gymnasium','numpy','pettingzoo','pygame','PyQt6','shapely','SuperSuit','torch','stable-baselines3']
def inspect(name):
    version=md.version(name)
    url=f'https://pypi.org/pypi/{name}/{version}/json'
    try:
        raw=urllib.request.urlopen(url,timeout=30).read()
    except urllib.error.HTTPError as error:
        if error.code==404:return name,{'version':version,'metadata_url':url,'published':False,'http_status':404}
        raise
    (out/f'{name}.json').write_bytes(raw)
    metadata=json.loads(raw)
    wheels=[]
    for item in metadata['urls']:
        if item['packagetype']!='bdist_wheel' or item['yanked']:continue
        if tags & parse_wheel_filename(item['filename'])[3]:
            wheels.append({key:item[key] for key in ('filename','url','size','digests')})
    return name,{'version':version,'metadata_url':url,'published':True,
                 'requires_python':metadata['info']['requires_python'],
                 'matching_wheels':wheels}
with ThreadPoolExecutor(max_workers=4) as pool:
    packages=dict(pool.map(inspect,names))
blue=packages['bluesky-simulator']
assert blue['version']=='1.1.1'
wheel=next(w for w in blue['matching_wheels'] if 'manylinux' in w['filename'])
path=out/wheel['filename']
assert not path.exists()
path.write_bytes(urllib.request.urlopen(wheel['url'],timeout=30).read())
assert hashlib.sha256(path.read_bytes()).hexdigest()==wheel['digests']['sha256']
installed=Path(md.distribution('bluesky-simulator').locate_file(''))
matched=[];different=[]
with zipfile.ZipFile(path) as archive:
    assert archive.testzip() is None
    compiled=[name for name in archive.namelist() if '/geo/_cgeo.' in name and name.endswith('.so')]
    assert len(compiled)==1
    elf=archive.read(compiled[0])
    assert elf[:6]==b'\x7fELF\x02\x01' and int.from_bytes(elf[18:20],'little')==62
    for name in archive.namelist():
        if name.startswith('bluesky/') and name.endswith('.py'):
            local=installed/name
            if local.is_file() and local.read_bytes().replace(b'\r\n',b'\n')==archive.read(name).replace(b'\r\n',b'\n'):
                matched.append(name)
            else:different.append(name)
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'target_assumption':'CPython 3.12, Linux x86-64 with glibc 2.35 or newer; actual university node OS/ABI remains unknown',
        'packages':packages,'downloaded_simulator_wheel':str(path.relative_to(root)),
        'simulator_wheel_sha256':wheel['digests']['sha256'],
        'compiled_geography_elf_x86_64':compiled[0],
        'matching_installed_python_files':len(matched),'different_python_files':different,
        'linux_code_executed':False,'dependencies_installed':False,
        'scope':'Published wheel availability and archive inspection; not a resolved dependency environment or proof of Linux runtime parity'}
(out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'packages':{n:{'version':v['version'],'published':v['published'],'matching_wheels':len(v.get('matching_wheels',[]))} for n,v in packages.items()},
                  'simulator_wheel_sha256':result['simulator_wheel_sha256'],
                  'compiled_geography':compiled[0],'matching_python_files':len(matched),'different_python_files':different,
                  'linux_code_executed':False},indent=2))
