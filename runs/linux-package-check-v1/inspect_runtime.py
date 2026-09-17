import sys,platform,importlib.metadata as md,json
names=['torch','numpy','bluesky-simulator','bluesky-navdata','openap','stable-baselines3','gymnasium','pettingzoo','supersuit','pygame','PyQt6','shapely']
packages={}
for name in names:
    try:packages[name]=md.version(name)
    except md.PackageNotFoundError:packages[name]=None
print(json.dumps({'python':sys.version,'machine':platform.machine(),'libc':platform.libc_ver(),'packages':packages},indent=2))
