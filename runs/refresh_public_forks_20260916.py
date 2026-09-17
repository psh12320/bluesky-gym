"""Refresh public fork metadata and known developed branch heads without authentication."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,urllib.request,urllib.parse
root=Path.cwd();out=root/'runs/public-refresh-20260916-v1';out.mkdir(exist_ok=False)
queries={'upstream':'https://api.github.com/repos/TUDelft-CNS-ATM/bluesky-gym',
         'forks':'https://api.github.com/repos/TUDelft-CNS-ATM/bluesky-gym/forks?per_page=100&sort=newest'}
owners=['CGCooke','danielkalmanson','cyf617','michael2992','Pablomg02']
for owner in owners:queries[owner]='https://api.github.com/repos/'+owner+'/bluesky-gym/branches?per_page=100'

def fetch(item):
    name,url=item
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'airtrafficcontrol-research','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(request,timeout=30) as response:
            body=response.read();headers={key:response.headers.get(key) for key in ('Date','ETag','Last-Modified','Link','X-RateLimit-Remaining')}
        return name,{'url':url,'status':'retrieved','body':body,'headers':headers}
    except Exception as error:return name,{'url':url,'status':'failed','error':repr(error)}
with ThreadPoolExecutor(max_workers=3) as pool:results=dict(pool.map(fetch,queries.items()))
audit={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Public API metadata only; no authenticated, private, email or Discord submissions accessed','requests':{}}
parsed={}
for name,item in results.items():
    body=item.pop('body',None)
    if body is not None:
        (out/f'{name}.json').write_bytes(body);item['sha256']=hashlib.sha256(body).hexdigest();parsed[name]=json.loads(body)
    audit['requests'][name]=item
if 'forks' in parsed:
    audit['listed_public_forks']=len(parsed['forks'])
    audit['fork_page_complete']=not results['forks']['headers'].get('Link')
    audit['forks']=[{'name':r['full_name'],'default_branch':r['default_branch'],'pushed_at':r['pushed_at']} for r in parsed['forks']]
if 'upstream' in parsed:audit['upstream_reported_fork_count']=parsed['upstream']['forks_count']
prior={'CGCooke':('AI4REAL-NET-Competition','a458870c711679e4fedb764abc1d58c8f4ffdb61'),
       'danielkalmanson':('daniel/main','55c636b933a8ad79fb9eb3821065daeba756e177'),
       'cyf617':('cursor/competition-mdp-v1-4e46','e3875b02857194d813272465a2311e8b8d5277d8')}
audit['prior_review_head_comparison']={}
for owner,(branch,digest) in prior.items():
    if owner not in parsed:continue
    current=next((b['commit']['sha'] for b in parsed[owner] if b['name']==branch),None)
    audit['prior_review_head_comparison'][owner]={'branch':branch,'prior_review_sha256_not_applicable_commit':digest,'current_commit':current,'unchanged':current==digest}
(out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k not in ('forks',)},indent=2))
