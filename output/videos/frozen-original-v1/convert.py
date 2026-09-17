"""Convert verified simulator reels to MP4 without running another simulation."""
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
out=root/'output/videos/frozen-original-v1'
ffmpeg=Path('C:/ffmpeg/bin/ffmpeg.exe')
ffprobe=ffmpeg.with_name('ffprobe.exe')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={'sa':root/'runs/preview-frozen-sa-reach250-v1',
        'ma':root/'runs/preview-fast-reference-interval5-v1-ma'}
manifest={'scope':'First five development scenarios for the selected frozen models with the original heading formatter. These videos do not represent the corrected deployment or population scores.',
          'ffmpeg_version':subprocess.check_output([str(ffmpeg),'-version'],text=True).splitlines()[0],
          'clips':{}}
for kind,folder in inputs.items():
    audit=json.loads((folder/'reel-audit.json').read_text())
    recording=json.loads((folder/'recording.json').read_text())
    source=folder/'five-scenario-reel.gif'
    target=out/f'{kind}-first-five.mp4'
    assert not target.exists()
    assert sha(source)==audit['reel_sha256']
    assert audit['episodes']==5 and audit['all_nine_metrics_exact_for_every_replay']
    command=[str(ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-n',
             '-ignore_loop','1','-i',str(source),'-vf','fps=10',
             '-c:v','libx264','-preset','medium','-crf','18','-threads','1',
             '-pix_fmt','yuv420p','-movflags','+faststart','-an',
             '-metadata',f'title={kind.upper()} first five development scenarios, original heading formatter',
             str(target)]
    subprocess.run(command,check=True)
    data=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    streams=data['streams']
    assert len(streams)==1 and streams[0]['codec_name']=='h264'
    assert (streams[0]['width'],streams[0]['height'])==tuple(audit['dimensions']) if 'dimensions' in audit else (720,828)==(streams[0]['width'],streams[0]['height'])
    duration=float(data['format']['duration'])
    expected=audit['duration_ms']/1000
    assert abs(duration-expected)<=.11,(duration,expected)
    subprocess.run([str(ffmpeg),'-v','error','-nostdin','-i',str(target),'-f','null','-'],check=True)
    seconds=duration-1
    frame=out/f'{kind}-final-check.png'
    subprocess.run([str(ffmpeg),'-v','error','-nostdin','-n','-ss',str(seconds),'-i',str(target),'-frames:v','1','-update','1',str(frame)],check=True)
    manifest['clips'][kind]={'source_gif_sha256':sha(source),'mp4_sha256':sha(target),
                           'model_sha256':recording['model_sha256'],
                           'decision_interval_seconds':recording['decision_interval_seconds'],
                           'episodes':[e['episode'] for e in recording['episodes']],
                           'file':target.name,'bytes':target.stat().st_size,'duration_seconds':duration,
                           'codec':'h264','pixel_format':streams[0]['pix_fmt'],
                           'full_video_decodes':True,'conversion_command':command,
                           'visual_check_frame':frame.name,'visual_inspection':'pending'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:{f:v for f,v in item.items() if f!='conversion_command'} for k,item in manifest['clips'].items()},indent=2))
