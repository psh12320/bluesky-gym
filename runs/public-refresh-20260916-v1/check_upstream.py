from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,urllib.request
root=Path.cwd();out=root/'runs/public-refresh-20260916-v1'
urls={'upstream-competition-branch':'https://api.github.com/repos/TUDelft-CNS-ATM/bluesky-gym/branches/AI4REAL-NET-Competition',
      'recent-issues':'https://api.github.com/repos/TUDelft-CNS-ATM/bluesky-gym/issues?state=all&sort=updated&direction=desc&per_page=30'}
def fetch(item):
    name,url=item
    request=urllib.request.Request(url,headers={'User-Agent':'airtrafficcontrol-research','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30) as response:body=response.read()
    path=out/f'{name}.json';assert not path.exists();path.write_bytes(body)
    return name,{'url':url,'sha256':hashlib.sha256(body).hexdigest(),'data':json.loads(body)}
with ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(fetch,urls.items()))
head=results['upstream-competition-branch']['data']['commit']['sha']
url=f'https://raw.githubusercontent.com/TUDelft-CNS-ATM/bluesky-gym/{head}/docs/competition/COMPETITION.md'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'airtrafficcontrol-research'}),timeout=30) as response:rules=response.read()
path=out/'competition-rules.md';assert not path.exists();path.write_bytes(rules)
local=(root/'docs/competition/COMPETITION.md').read_text()
audit={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'upstream_competition_commit':head,
       'matches_frozen_upstream_commit':head=='00930013219af4c17e509c3efc84a8becd0f3546',
       'rules_url':url,'rules_sha256':hashlib.sha256(rules).hexdigest(),
       'rules_equal_to_local_after_newline_normalization':rules.decode().replace('\r\n','\n')==local.replace('\r\n','\n'),
       'requests':{name:{k:v for k,v in item.items() if k!='data'} for name,item in results.items()},
       'recent_issue_titles':[{'number':i['number'],'title':i['title'],'updated_at':i['updated_at'],'url':i['html_url'],'is_pull_request':'pull_request' in i} for i in results['recent-issues']['data']]}
(out/'upstream-refresh-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
