"""Build an MP4 from the verified corrected-deployment development recordings."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from PIL import Image,ImageSequence
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--env',choices=['sa','ma'],required=True)
args=parser.parse_args()
folder=Path(__file__).resolve().parent/args.env
recording=json.loads((folder/'recording.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert recording['recording_complete'] and recording['recording_error'] is None
assert recording['heading_transport']=={'revision':1,'encoding':'plain_decimal'}
assert recording['seed']==2026 and [e['episode'] for e in recording['episodes']]==list(range(5))
frames,durations=[],[]
for item in recording['episodes']:
    path=folder/item['file']
    assert sha(path)==item['gif_sha256']
    assert item['matches_reference'] and item['scenario_rng_unchanged']
    assert all(value<=(1e-5 if key=='total_reward' else 0) for key,value in item['maximum_absolute_metric_differences'].items())
    with Image.open(path) as gif:
        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.convert('RGB'))
            durations.append(frame.info['duration'])
reel=folder/'five-scenario-reel.gif'
assert not reel.exists()
frames[0].save(reel,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False,disposal=2)
count=len(frames)
frames.clear()
with Image.open(reel) as gif:
    assert gif.n_frames==count and gif.size==(720,828)
    assert sum(frame.info['duration'] for frame in ImageSequence.Iterator(gif))==sum(durations)
out=ROOT/'output/videos/decimal-heading-v1'
out.mkdir(parents=True,exist_ok=True)
video=out/f'{args.env}-first-five.mp4'
assert not video.exists()
ffmpeg=Path('C:/ffmpeg/bin/ffmpeg.exe')
command=[str(ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-n',
         '-ignore_loop','1','-i',str(reel),'-vf','fps=10','-c:v','libx264',
         '-preset','medium','-crf','18','-threads','1','-pix_fmt','yuv420p',
         '-movflags','+faststart','-an','-metadata',
         f'title={args.env.upper()} first five development scenarios, decimal heading transport revision 1',str(video)]
subprocess.run(command,check=True)
probe=json.loads(subprocess.check_output([str(ffmpeg.with_name('ffprobe.exe')),'-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
assert len(probe['streams'])==1
stream=probe['streams'][0]
assert stream['codec_name']=='h264' and (stream['width'],stream['height'])==(720,828)
duration=float(probe['format']['duration'])
assert abs(duration-sum(durations)/1000)<=.11
subprocess.run([str(ffmpeg),'-v','error','-nostdin','-i',str(video),'-f','null','-'],check=True)
checks={}
for label,seconds in [('motion',min(1.5,duration/3)),('final',duration-1)]:
    frame=out/f'{args.env}-{label}-check.png'
    subprocess.run([str(ffmpeg),'-v','error','-nostdin','-n','-ss',str(seconds),'-i',str(video),'-frames:v','1','-update','1',str(frame)],check=True)
    checks[label]=frame.name
old=ROOT/('runs/preview-frozen-sa-reach250-v1/five-scenario-reel.gif' if args.env=='sa'
          else 'runs/preview-fast-reference-interval5-v1-ma/five-scenario-reel.gif')
audit={'env':args.env,'episodes':5,'seed':2026,'heading_transport':recording['heading_transport'],
       'model_sha256':recording['model_sha256'],'deployment_manifest_sha256':recording['deployment_manifest_sha256'],
       'recording_sha256':sha(folder/'recording.json'),'reel_sha256':sha(reel),'reel_frame_count':count,
       'matches_original_reel_bytes':sha(reel)==sha(old),'file':video.name,'mp4_sha256':sha(video),
       'duration_seconds':duration,'bytes':video.stat().st_size,'codec':stream['codec_name'],
       'full_stream_decodes':True,'visual_check_frames':checks,'visual_inspection':'pending',
       'all_nine_metrics_exact':all(v==0 for e in recording['episodes'] for v in e['maximum_absolute_metric_differences'].values()),
       'scope':'First five development scenarios; corrected deployment, not a population estimate or official-protocol score',
       'conversion_command':command}
(out/f'{args.env}-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='conversion_command'},indent=2))
