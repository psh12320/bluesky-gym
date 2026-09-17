from pathlib import Path
import hashlib,json
from PIL import Image,ImageSequence
p=Path('runs/preview-frozen-sa-reach250-v1')
r=json.loads((p/'recording.json').read_text())
assert r['model_sha256']=='f89ea9427a6dc2900286bbe94f39119a150ec2bb10bb589333f8c89c32349f08'
assert r['decision_interval_seconds']==10 and r['seed']==2026
assert [e['episode'] for e in r['episodes']]==list(range(5))
frames,durations=[],[]
for e in r['episodes']:
    f=p/e['file']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==e['gif_sha256']
    assert e['matches_reference'] and e['scenario_rng_unchanged']
    assert all(v==0 for v in e['maximum_absolute_metric_differences'].values())
    with Image.open(f) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert('RGB'))
            durations.append(frame.info['duration'])
out=p/'five-scenario-reel.gif'
assert not out.exists()
frames[0].save(out,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False,disposal=2)
with Image.open(out) as im:
    assert im.n_frames==len(frames) and im.size==(720,828)
    assert sum(frame.info['duration'] for frame in ImageSequence.Iterator(im))==sum(durations)
    im.seek(75)
    im.convert('RGB').save(p/'reel-middle-check.png')
audit={'episodes':5,'frame_count':len(frames),'duration_ms':sum(durations),'dimensions':[720,828],'all_nine_metrics_exact_for_every_replay':True,'all_individual_gif_hashes_verified':True,'model_sha256':r['model_sha256'],'decision_interval_seconds':10,'reel_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'reel_bytes':out.stat().st_size,'scope':'First five development scenarios; frozen SA candidate; not a population estimate or official scoring run'}
(p/'reel-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
