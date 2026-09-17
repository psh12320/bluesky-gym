"""Build a development-only report from verified evaluation records."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
PDF = OUT/'feasibility-working-draft.pdf'
SNAPSHOT = Path(__file__).resolve().parent/'evidence.json'
METRICS = ['waypoint_reached','flight_time','intrusion_events','intrusion_time','restricted_area_events','time_in_restricted_area','sector_exit_events','time_outside_sector','total_reward']
PREFIXES = {
 'ma_classical':'runs/goal-route-choice-v1-fast-joint-ma-200',
 'ma_historical':'runs/sac-ma-public-600k-seed1400/validation-individual-200',
 'ma_learned':'runs/sac-ma-fast-reference-v1-25k-seed2900/validation-200',
 'sa_classical':'runs/goal-route-choice-v1-fast-joint-sa-200',
 'sa_learned':'runs/fast-reference-reach250-v1-ma25k-sa/validation-200',
}
evidence={}
for name,prefix in PREFIXES.items():
    path=ROOT/prefix
    meta=json.loads(path.with_suffix('.json').read_text())
    with path.with_suffix('.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    count=2000 if name.startswith('ma') else 200
    assert meta['seed']==2026 and meta['episodes']==200 and len(rows)==count
    assert len({(r['episode'],r['agent']) for r in rows})==count
    assert {int(r['episode']) for r in rows}==set(range(200))
    for metric in METRICS:
        actual=math.fsum(float(r[metric]) for r in rows)/len(rows)
        assert math.isclose(actual,meta['metrics'][metric]['mean'],abs_tol=1e-9,rel_tol=1e-12),(name,metric)
    clean=sum(float(r['waypoint_reached'])==1 and all(float(r[k])==0 for k in METRICS[2:-1]) for r in rows)/len(rows)
    assert math.isclose(clean,meta['clean_completion_rate'],abs_tol=1e-12)
    evidence[name]={'prefix':prefix,'csv_sha256':hashlib.sha256(path.with_suffix('.csv').read_bytes()).hexdigest(),
                    'json_sha256':hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),'metadata':meta}
assert evidence['ma_learned']['metadata']['model_sha256']=='961f525f11bec7a0f417933d209230ca99c5b4b8c00d883233a006b8ee38c4bb'
assert evidence['sa_learned']['metadata']['model_sha256']=='f89ea9427a6dc2900286bbe94f39119a150ec2bb10bb589333f8c89c32349f08'
SNAPSHOT.write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Development only; held-out outcome files are not read','evaluations':evidence},indent=2),encoding='utf-8')

ink=colors.HexColor('#172033')
blue=colors.HexColor('#174d79')
styles={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=ink,spaceAfter=12),
 'subtitle':ParagraphStyle('subtitle',fontName='Helvetica',fontSize=12,leading=15,textColor=blue,spaceAfter=15),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=14,leading=17,textColor=blue,spaceBefore=9,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=12,leading=15,spaceAfter=8,textColor=ink),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=12,leading=14,textColor=ink),
 'ref':ParagraphStyle('ref',fontName='Helvetica',fontSize=11,leading=13.5,spaceAfter=7,textColor=ink),
}
story=[]
def p(text,kind='body'):
    story.append(Paragraph(text,styles[kind]))
def h(text):
    p(text,'head')
def table(track):
    labels=[('waypoint_reached','Arrival (%)',True),('flight_time','Flight time (s)',False),
            ('intrusion_events','Intrusion events',False),('intrusion_time','Intrusion time (s)',False),
            ('restricted_area_events','Restricted-area events',False),('time_in_restricted_area','Restricted-area time (s)',False),
            ('sector_exit_events','Sector-exit events',False),('time_outside_sector','Outside-sector time (s)',False)]
    a=evidence[track+'_classical']['metadata']; b=evidence[track+'_learned']['metadata']
    data=[['Metric','Classical','Learned']]
    for key,label,percent in labels:
        va,vb=a['metrics'][key]['mean'],b['metrics'][key]['mean']
        data.append([label,f'{va*100:.2f}' if percent else f'{va:.4f}',f'{vb*100:.2f}' if percent else f'{vb:.4f}'])
    data.append(['Clean arrival (%)',f"{100*a['clean_completion_rate']:.2f}",f"{100*b['clean_completion_rate']:.2f}"])
    data=[[Paragraph(escape(str(x)),styles['cell']) for x in row] for row in data]
    t=Table(data,colWidths=[221,105,113],hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef5')),('LINEBELOW',(0,0),(-1,0),.6,blue),
      ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
      ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')])]))
    story.append(t)

p('Residual learning for<br/>air traffic control','title')
p('Feasibility draft | Development evidence | 15 Sep 2026','subtitle')
h('Study question')
p('Can a learned correction improve a strong route follower while preserving arrivals and reducing airspace violations? We combine shared Soft Actor-Critic (SAC), geometric guidance and a joint command filter in BlueSky-Gym [1-3]. MA clean arrival rises from 27.7% for an earlier unfiltered learner to 94.1%, with different training budgets. A clear overall gain over matched classical control is unproven. Held-out and official results are pending.')
h('Problem and evaluation boundary')
p('The competition uses procedural sectors with five obstacles [2]. SA controls one A320 among ten scripted intruders; MA controls ten aircraft. Separation is 5 NM, goal capture is 5 km, and the horizon is 3000 s. One-second scoring, scenarios, dynamics and termination remain unchanged. Control is horizontal; built-in conflict resolution is off.')
h('Controller')
p('A visibility planner considers obstacle clearances of 6, 3, 2, 1 and 0 km. It prefers paths with at least 2 km clearance and selects the widest path within 15% of the shortest preferred alternative. The route bearing replaces the direct bearing in a 124-element observation. A shared SAC actor adds at most 15 degrees to that reference; the composed heading command is limited to 45 degrees. Its speed command is clip(1 + 2u, -1, 1), so zero residual reproduces the fast classical reference.')
p('The joint filter keeps a predicted-clear nominal command. Otherwise it searches heading/speed candidates under a 90 s forecast, sampled every 5 s with segment-level separation checks. Aircraft disappear from predictions after goal capture. A stable aircraft order shares previously chosen plans. If no jointly feasible candidate exists, static feasibility is prioritized and predicted conflict risk is minimized within the remaining candidates. If every candidate violates static constraints, a static-exposure cost takes priority. Finite search and approximate dynamics prevent a safety guarantee.')
h('Learning and attribution')
p('Residual learning and initialization follow established work [4,5]. Reward/discount parameters come from a public fork [6], without its Rust collector or trained weights. Novelty must lie in the ATC method and evidence; neither residual RL nor public-score reproduction is claimed as our contribution.')
story.append(PageBreak())

h('Completed development results')
p('Each table uses the same 200 seed-2026 scenarios per track, seeded once and then continued. MA has 2000 aircraft records; SA has 200. Values are means per aircraft. Clean arrival means a reached goal with zero traffic, obstacle and sector violations. Reward totals are excluded because the reward definitions differ.')
h('MA: ten-second fast-reference residual')
table('ma')
h('SA: arrival-reward residual transfer')
table('sa')
p('Both learned models use 25,000 counted MA transitions. The SA model changes the arrival reward to 250 and transfers without SA training. It still simulates ten intruders while observing nine slots. MA five-second control remains a separate pending expansion; it is not substituted into this completed table.')
story.append(PageBreak())

h('What the comparisons establish')
p('MA changes flight by -5.77 s (95% paired interval -18.51 to 7.02), intrusion by +0.142 s (-0.584 to 0.906), and arrival by -0.15 percentage points (-0.60 to 0.25) versus classical. The intervals do not establish an overall learned gain. The SA arrival-reward model completes all goals but lengthens flight by 45.77 s (11.65 to 78.13); its intrusion change of -1.005 s (-4.530 to 2.415) is uncertain.')
figure=ROOT/'runs/report-draft-v1/sa-distributions.png'
story.append(Image(str(figure),width=451,height=451*2.8/6.26))
p('<b>Figure 1.</b> SA development distributions. The arrival-reward change removes a timeout while shifting flight durations upward. Its conflict curve crosses the classical curve, so a smaller average does not establish uniform safety improvement. Curves are descriptive, without confidence intervals. The companion tail analysis includes MA and static violations.')
h('Ablations and failure analysis')
p('Increasing the MA budget to 100k transitions does not improve the full twenty-scenario outcome: flights lengthen and one aircraft incurs 95 s of restricted exposure. Raising the arrival reward also loses two MA arrivals at 25k. These variants are retained as negative or mixed results. An independent fast-reference seed completes all twenty pilot arrivals on both tracks, but safety varies across seeds.')
p('The original SA learner misses development scenario 28 after 3000 s with 41 s intrusion. The arrival-reward learner reaches it in 885 s with 142 s intrusion; classical reaches it in 888 s with 217 s intrusion. The recovery therefore has a safety cost relative to the timeout. This case motivated the reward ablation and is not an unseen result.')
story.append(PageBreak())

h('Reproducibility and remaining evidence')
p('SAC uses 64x64 networks, learning rate 0.0003, tau 0.005, discount 0.99650847, batch 256, replay capacity 100,000, 5000 initial collection transitions and four updates per ten counted MA transitions. Counted transitions include inactive-aircraft padding; live counts are saved separately. The evaluated 25k callbacks contain 7996 updates, preceding the final four updates. Deterministic inference is per aircraft, matching the original harness call pattern.')
p('Model/configuration/source hashes, raw CSVs and paired comparisons are preserved. Bootstrap resampling uses whole scenarios and reports pointwise 95% intervals; it does not include training-seed uncertainty. The 89-test suite passes, and original-harness development checks reproduce all nine metrics for tested prefixes. Five-scenario previews also reproduce saved scores exactly.')
p('The SA arrival-reward candidate is frozen for 200 unseen seed-2027 scenarios against classical control. Its source and configuration were recorded before rollouts. MA selection uses only its development comparisons. No held-out outcome appears in this draft, and no official seed-42 run is complete. A competition submission still requires the original 1000-scenario harness, a final checkpoint decision and a reviewed report/video package.')
h('Responsible interpretation')
p('Mean safety scores can conceal severe individual failures: one MA aircraft spends 101 s outside the sector despite only 0.0505 s mean exposure. Starting violations remain included. Simulation slots are not demographic groups, and flight duration is not route-normalized delay. No operational safety, fairness or real-world emissions claim is established. Team contributions and the course-required tool-use declaration must be completed accurately before any submission.')
h('References')
refs=[
 '[1] Groot DJ, Leto G, Vlaskin A, Moec AAG, Ellerbroek J. BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications. SESAR Innovation Days; 2024. <link href="https://research.tudelft.nl/en/publications/bluesky-gym-reinforcement-learning-environments-for-air-traffic-a/" color="#174d79">TU Delft record</link>.',
 '[2] TUDelft-CNS-ATM. BlueSky-Gym competition specification. Revision 0093001, AI4REAL-NET-Competition branch. <link href="https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/00930013219af4c17e509c3efc84a8becd0f3546/docs/competition/COMPETITION.md" color="#174d79">Specification</link>.',
 '[3] Haarnoja T, Zhou A, Abbeel P, Levine S. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. PMLR. 2018;80:1861-1870. <link href="https://proceedings.mlr.press/v80/haarnoja18b.html" color="#174d79">Paper</link>.',
 '[4] Johannink T, Bahl S, Nair A, Luo J, Kumar A, Loskyll M, et al. Residual Reinforcement Learning for Robot Control. arXiv:1812.03201; 2018. <link href="https://arxiv.org/abs/1812.03201" color="#174d79">Preprint</link>.',
 '[5] Silver T, Allen K, Tenenbaum J, Kaelbling L. Residual Policy Learning. arXiv:1812.06298; 2018. <link href="https://arxiv.org/abs/1812.06298" color="#174d79">Preprint</link>.',
 '[6] CGCooke. BlueSky-Gym competition experiments and parameter reference. Revision a458870c, reviewed 15 Sep 2026. <link href="https://github.com/CGCooke/bluesky-gym/tree/a458870c711679e4fedb764abc1d58c8f4ffdb61" color="#174d79">Repository snapshot</link>.',
]
for ref in refs:p(ref,'ref')

def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c9d4df'))
    canvas.line(72,48,A4[0]-72,48)
    canvas.setFont('Helvetica',9)
    canvas.setFillColor(colors.HexColor('#516173'))
    canvas.drawString(72,34,'Working draft | Development results only | Not a submission')
    canvas.drawRightString(A4[0]-72,34,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=72,leftMargin=72,topMargin=72,bottomMargin=72,title='Residual learning for air traffic control - feasibility working draft',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(PDF)
record={'pdf':str(PDF),'pages':len(reader.pages),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'evidence_sha256':hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'verified_development_evaluations':list(evidence),'heldout_outcomes_read':False,'official_results_claimed':False}
(Path(__file__).resolve().parent/'build.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
