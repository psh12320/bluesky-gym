"""Advise Linux to release clean cache pages for older files in this task's runtime."""
import json,os,time
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2]/'runs/linux-runtime-v1';root=root.resolve()
assert root.name=='linux-runtime-v1' and root.parent.name=='runs'
assert hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED')
output=root/'cache-advice.json';assert not output.exists()
def memory():return Path('/proc/meminfo').read_text()
before=memory();cutoff=time.time()-60;advised=[];errors=[]
for folder in (root/'wheelhouse-v2',root/'python',root/'.venv'):
    for path in folder.rglob('*'):
        try:
            if path.is_symlink() or not path.is_file():continue
            info=path.stat()
            if info.st_size<65536 or info.st_mtime>cutoff:continue
            resolved=path.resolve();assert resolved.is_relative_to(root)
            with resolved.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            advised.append({'path':str(resolved.relative_to(root)),'bytes':info.st_size})
        except OSError as error:errors.append({'path':str(path),'error':repr(error)})
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':str(root),
        'operation':'POSIX_FADV_DONTNEED on read-only handles to task-owned regular files at least 64 KiB and older than 60 seconds',
        'file_contents_modified':False,'advisory_only':True,'advised_files':advised,'errors':errors,
        'meminfo_before':before,'meminfo_after':memory()}
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'advised_files':len(advised),'advised_bytes':sum(x['bytes'] for x in advised),
                  'errors':len(errors),'file_contents_modified':False,'meminfo_before':before,'meminfo_after':result['meminfo_after']},indent=2))
