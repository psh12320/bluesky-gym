"""Build a four-page feasibility report only from the completed evidence snapshot."""
from pathlib import Path
from xml.sax.saxutils import escape
from datetime import datetime, timezone
import hashlib
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'evidence.json'
if not SOURCE.is_file():
    raise SystemExit('Complete collect_evidence.py before generating the final report')
E = json.loads(SOURCE.read_text(encoding='utf-8'))
assert E['replications']['all_declared_seeds_included']
for track in ('sa', 'ma'):
    assert E[f'corrected_{track}_full_score']['summary']['episodes'] == 1000
OUT = ROOT / 'output/pdf/feasibility-evaluation.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
INK, BLUE = colors.HexColor('#172033'), colors.HexColor('#174d79')
WIDTH = A4[0] - 144
styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=22, leading=25, textColor=INK, spaceAfter=10),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=12, leading=15, textColor=BLUE, spaceAfter=10),
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=BLUE, spaceBefore=8, spaceAfter=6),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=12, leading=14.5, textColor=INK, spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=12, leading=14, textColor=INK),
    'ref': ParagraphStyle('ref', fontName='Helvetica', fontSize=12, leading=14, textColor=INK, spaceAfter=6),
}
story = []
def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))
def h(text):
    p(text, 'head')
def table(data, widths):
    rows = [[Paragraph(escape(str(value)), styles['cell']) for value in row] for row in data]
    t = Table(rows, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eef5')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, BLUE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9fb')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 7))
def value(summary, metric):
    return summary['metrics'][metric]['mean'] if metric in summary['metrics'] else summary[metric]
def fmt(v, percent=False):
    return f'{100*v:.2f}%' if percent else f'{v:.4f}'

p('Residual learning for<br/>air traffic control', 'title')
p('Local feasibility evaluation | ' + E['collected_at_utc'][:10], 'subtitle')
h('Question and current conclusion')
p('Can a learned correction improve a strong route follower in procedural airspaces? The working system combines shared Soft Actor-Critic (SAC), geometric guidance and a joint command filter [1-3]. The matched comparisons below do not establish an overall learned MA advantage. SA arrival results are more promising, but vary across training seeds. These results support continued targeted research; they do not establish a competition ranking or a winning method.')
h('Task and controller')
p('SA controls one A320 among ten scripted intruders; MA controls ten aircraft. Horizontal heading and speed control must preserve 5 NM separation, avoid restricted areas and remain inside the sector. Goal capture is 5 km and the horizon is 3000 s. Original one-second scoring, dynamics, scenario construction and termination are unchanged; built-in conflict resolution is off [1].')
p('A visibility planner tries obstacle clearances of 6, 3, 2, 1 and 0 km. It prefers at least 2 km clearance and selects the widest route within 15% of the shortest preferred alternative. The route bearing enters a 124-element observation. A shared actor adds up to 15 degrees of residual heading, with the composed turn command clipped to 45 degrees. Its speed action is clip(1 + 2u, -1, 1). Zero residual reproduces the fast classical reference.')
p('The filter forecasts 90 s ahead in 5 s segments, including aircraft disappearance after goal capture. It accepts a predicted-clear command or searches alternative headings and speeds. Aircraft are processed in stable order using earlier chosen plans. If no joint solution exists, static feasibility takes priority; if no static solution exists, predicted static exposure is minimized first. This finite search and approximate forecast provide no safety guarantee. The filter is centralized although the learned policy is shared.')
h('Training and attribution')
p('SAC uses two 64-unit hidden layers, batch size 256 and four updates per vector step. Each selected callback has 25,000 counted transitions and 7,996 optimizer updates; padding is reported separately. Training decisions occur every 10 s. Deployment uses 5 s for MA and 10 s for SA. SA uses MA training with arrival reward 250, without SA training. Residual learning is established work [3]; public reward and discount choices are attributed to [4], without claiming its collector or weights.')
story.append(PageBreak())

h('Full competition-protocol evaluation')
p('Each selected deployment ran 1000 scenarios from seed 42, seeded once and then continued. SA yields 1000 aircraft records; MA yields 10,000. These are local Windows CPU results with compiled BlueSky geometry and OpenAP, not organizer-verified scores. All rows, including failed arrivals, are retained. Values below are means per aircraft; event counts remain separate from exposure time.')
labels = [
    ('waypoint_reached', 'Arrival', True), ('flight_time', 'Flight time (s)', False),
    ('intrusion_events', 'Intrusion events', False), ('intrusion_time', 'Intrusion time (s)', False),
    ('restricted_area_events', 'Restricted events', False), ('time_in_restricted_area', 'Restricted time (s)', False),
    ('sector_exit_events', 'Sector-exit events', False), ('time_outside_sector', 'Outside time (s)', False),
    ('clean_completion_rate', 'Clean arrival', True), ('all_aircraft_clean_completion_rate', 'All-aircraft clean', True),
]
full = {track:E[f'corrected_{track}_full_score']['summary'] for track in ('sa', 'ma')}
data = [['Metric', 'SA', 'MA']]
for key, label, percent in labels:
    data.append([label, *[fmt(value(full[t], key), percent) for t in ('sa', 'ma')]])
table(data, [WIDTH-190, 95, 95])
p('Clean arrival requires reaching the goal with zero traffic, restricted-area and sector violations. All-aircraft clean requires every controlled aircraft in a scenario to satisfy this condition. Reward is not an official aggregate score and is not compared across reward definitions; every raw reward record is included in the saved evaluation CSVs.')
h('Failure retention and deployment verification')
misses = {t:round(full[t]['agent_episodes']*(1-value(full[t], 'waypoint_reached'))) for t in ('sa', 'ma')}
p(f"The full runs retain {misses['sa']} SA and {misses['ma']} MA missed arrivals. Scenario-level failures and exposure tails are available in SCORING_RESULTS.md and the full CSVs. Mean safety values alone are insufficient to establish reliability.")
transport = E['sa_transport_revision_comparison']
if all(v['changed_aircraft_records'] == 0 for v in transport.values()):
    p('The corrected decimal heading transport reproduces all nine metrics of every original-format SA aircraft record exactly. The correction addresses a separately reproduced scientific-notation command-parser error in MA; a single-case replay is not evidence of full MA trajectory equivalence.')
else:
    changed = ', '.join(k for k,v in transport.items() if v['changed_aircraft_records'])
    p('The SA transport revision comparison reports changed records in: ' + escape(changed) + '. The complete per-metric comparison is retained in the evidence snapshot; full-run equivalence is not claimed.')
p('The packaged deployment passed 108 regression tests. Fresh extraction and five-scenario previews reproduce saved metrics on Windows. Native Linux execution remains unverified; no university Slurm job has been submitted.')
story.append(PageBreak())

h('Does learning add value?')
p('The primary learned and matching classical controllers were compared on the same 200 seed-2027 scenarios per track. Candidates were selected before those outcomes. Replicas subsequently reused this stream; they are not a new untouched test. These comparisons use the original heading formatter and are distinct from corrected full scoring.')
data = [['Metric', 'SA class.', 'SA learned', 'MA class.', 'MA learned']]
for key,label,percent in [
    ('waypoint_reached','Arrival',True), ('flight_time','Flight (s)',False),
    ('intrusion_time','Intrusion (s)',False), ('time_in_restricted_area','Restricted (s)',False),
    ('time_outside_sector','Outside (s)',False), ('clean_completion_rate','Clean arrival',True),
]:
    data.append([label,*[fmt(value(E['heldout'][t][arm]['summary'],key),percent) for t,arm in [('sa','classical'),('sa','learned'),('ma','classical'),('ma','learned')]]])
table(data,[WIDTH-300,75,75,75,75])
p('Primary SA recovers six arrivals: +3 percentage points, with a paired 95% scenario-bootstrap interval of +1 to +5.5 points. Its safety and flight differences remain uncertain. Primary MA has the same arrival rate as classical control; every nontrivial paired interval includes zero. It recovers three classical misses and adds three. These results do not establish overall learned superiority.')
h('All declared training seeds')
data = [['Training seed','SA arrival','MA arrival','SA clean','MA clean']]
for seed in ('2900','2902','2904'):
    sa=E['replications']['tracks']['sa']['training_seeds'][seed]['values']
    ma=E['replications']['tracks']['ma']['training_seeds'][seed]['values']
    data.append([seed+(' (primary)' if seed=='2900' else ''),fmt(sa['waypoint_reached'],True),fmt(ma['waypoint_reached'],True),fmt(sa['clean_completion_rate'],True),fmt(ma['clean_completion_rate'],True)])
table(data,[WIDTH-300,75,75,75,75])
p('Every declared seed is included; the original primary models remain selected. Full per-seed event counts, exposure, flight time and paired comparisons are retained in REPLICATIONS.md and per-seed.csv. Scenario-bootstrap intervals are pointwise and conditional on one trained model. Three training seeds support descriptive mean, range and sample standard deviation, not precise population-level training uncertainty. Reusing the same 200 worlds does not create 600 independent scenarios.')
story.append(PageBreak())

h('Decision, limitations and next work')
p('A functioning controller, saved models, reproducible metrics and demonstration videos are available. The main research risk is whether learning contributes enough beyond the strong planner and filter. Most of the observed safety improvement over the historical unfiltered learner can be explained by changes outside the policy; that historical comparison also has different training budgets. A competitive claim requires a consistent gain over a matching strong baseline, with arrivals, safety, delay and adverse outcomes reported together.')
p('Additional development tests did not justify replacing the selected deployment. Waypoint-region routing shortened one aircraft flight by 126 s across 20 MA worlds: only 0.63 s per aircraft, with identical arrival and safety records. Batched forecast kernels matched scalar trajectories, but two paired end-to-end timing comparisons changed in opposite directions under varying load. Neither result establishes a sizeable general improvement.')
p('Public fork results are useful targets, not a verified leaderboard. The latest bounded check enumerated 38 public forks while repository metadata reported 39; the discrepancy is unresolved. The three previously inspected implementation heads were unchanged. Different or incompletely verified evaluation conditions prevent direct ranking against public self-reports [4].')
p('Next work should target a measurable learned contribution, followed by a new predeclared evaluation stream and native Linux reproduction. GPU availability alone does not resolve simulator throughput or memory limits. Cluster partition/account details and interactive authentication remain pending. Operational safety and causal fairness have not been established.')
p('The course option remains conditional on the team decision. Its proposal, presentation, contribution statements and final report are separate deliverables. Actual member contributions, assistance disclosure and reference-manager formatting require truthful team review; this report does not claim those steps are complete. Competition submission additionally requires organizer review and external submission, neither of which has occurred.')
h('References')
p('[1] TUDelft-CNS-ATM. BlueSky-Gym AI4REAL-NET competition rules. Competition branch, commit 00930013219af4c17e509c3efc84a8becd0f3546. <link href="https://github.com/TUDelft-CNS-ATM/bluesky-gym/tree/AI4REAL-NET-Competition" color="#174d79">Repository and rules</link>.', 'ref')
p('[2] Haarnoja T, Zhou A, Abbeel P, Levine S. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. PMLR. 2018;80:1861-1870. <link href="https://proceedings.mlr.press/v80/haarnoja18b.html" color="#174d79">Primary paper</link>.', 'ref')
p('[3] Johannink T, Bahl S, Nair A, et al. Residual Reinforcement Learning for Robot Control. 2018. arXiv:1812.03201. <link href="https://arxiv.org/abs/1812.03201" color="#174d79">Primary paper</link>.', 'ref')
p('[4] CGCooke. Public BlueSky-Gym competition implementation and reported experiments. Commit a458870c711679e4fedb764abc1d58c8f4ffdb61. <link href="https://github.com/CGCooke/bluesky-gym/tree/a458870c711679e4fedb764abc1d58c8f4ffdb61" color="#174d79">Pinned source</link>.', 'ref')

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica',9)
    canvas.setFillColor(BLUE)
    canvas.drawString(72,40,'Feasibility evidence | Local evaluation')
    canvas.drawRightString(A4[0]-72,40,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=72,rightMargin=72,topMargin=65,bottomMargin=60,
                      title='Residual learning for air traffic control: feasibility evaluation',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT)
assert len(reader.pages)==4,f'Expected four pages, found {len(reader.pages)}; inspect and revise layout'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
qa={'created_at_utc':datetime.now(timezone.utc).isoformat(),'pages':len(reader.pages),
    'evidence_sha256':sha(SOURCE),'builder_sha256':sha(Path(__file__)),'pdf_sha256':sha(OUT),
    'visual_review_complete':False,'scope':'Local feasibility report; not externally submitted; attribution requires team review'}
(BASE/'report-build.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))
