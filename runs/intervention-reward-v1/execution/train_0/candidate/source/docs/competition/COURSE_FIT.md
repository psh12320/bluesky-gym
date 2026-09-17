# Course fit and decision requirements

Reviewed 15 September 2026 against the supplied CS4246/5446 Project Guidelines
(Draft), issued 26 August 2026. The PDF is reference material; this note does not
change the team's current choice to build and evaluate first, then decide whether
to use this work for the course. No proposal or course submission has been made by
this worktree.

## Fit

The best fit is **Competition Project**, printed as item 4 on page 2. It requires
a working AI controller in a suitable planning/game/robotics environment; formal
competition entry is explicitly unnecessary (also page 4, footnote 3). BlueSky's
Gymnasium/PettingZoo interfaces and the implemented SAC controller fit this
interpretation. This is a reasoned mapping to the guidelines, not instructor
approval or a promise of a grade. Application Project is a possible alternative,
but no track change is needed to continue the current feasibility sprint.

Multi-Agent Systems is explicitly listed as a suggested direction. The course's
responsible-AI emphasis can be addressed through safety/efficiency tradeoffs,
transparent failure analysis, reproducibility and the distribution of delays and
safety failures among aircraft. Current simulation results do not demonstrate
operational air-traffic-control suitability or real-world fuel/emissions savings.

The user reports six team members and completed Canvas registration. This matches
the required team size of 5-6; registration is user-confirmed, not independently
checked in Canvas.

## Deadlines and submission formats

| Deliverable | Supplied draft requirement | Current state / remaining evidence |
| --- | --- | --- |
| Proposal | 28 September 2026, 18:00. Mandatory to receive a grade, although ungraded. Maximum one content page, single-spaced, single-column, at least 10 pt, 1-inch margins; title page and references excluded. | Not prepared or submitted. Our 15-24 September sprint leaves four days for the decision and proposal. |
| Proposal content | Background, problem statement, approach, team roles, deadlines and risk assessment. Canvas submission includes every member's name, matriculation number and email. | Experimental background and risks exist. Team identities, assigned roles and submission confirmation are not available. |
| Presentation | Week of 10 November 2026. About 15 minutes plus 5 minutes of questions, subject to change; in class or recorded if required. | Working previews exist. A presentation and shared team understanding still need preparation. |
| Report | Draft literally prints 23:59 Sunday, 15 November 2025. Academic technical PDF; maximum ten content pages, single-spaced, single-column, at least 12 pt, 1-inch margins. Title page/references excluded; important discussions and figures belong in the main report. | The printed year conflicts with AY2026/27; Sunday 15 November occurs in 2026. Use mid-November 2026 for planning only and confirm the actual Canvas deadline. No final report exists. |
| References | Use a reference manager and Vancouver citation style. Cite sources properly; do not copy or merely paraphrase source material. | Public-work sources, a numbered draft bibliography and references.ris are recorded. Reference-manager import and final formatting remain pending. |
| Contributions | Define team roles in proposal and report, state individual contributions, and ensure all members understand the project. Specify human contributions versus AI-generated content when using LLMs. | Record actual work and assistance accurately. No invented member assignments, claimed reviews, or completed authorship declaration. |

Source locations: dates/team/themes on page 1; types and compulsory proposal on
page 2; formats, citations and collaboration on page 3; grading and competition
entry clarification on page 4; detailed rubric on page 5. The PDF has seven pages.
The apparently irregular project-type numbering is reproduced by label rather
than repaired silently.

## Evidence already useful for assessment

- Problem formulation: two-dimensional heading/speed control, procedural sectors,
  conflict avoidance, restricted areas, arrivals and finite-horizon efficiency.
- Implementation: trained SAC models, shared MA policy, zero-shot SA transfer,
  geometric guidance and command filtering, configuration validation and replay.
- Methodology: paired scenarios, complete physical metrics, classical and trained
  baselines, preserved failed experiments, model/source hashes and simulator checks.
- Demonstration: five-scenario MA and SA previews with exact evaluation replay.
- Analysis: reward/arrival tradeoffs, training-seed variation, control timing and
  cases where classical guidance explains most of the improvement.

These support the rubric's problem understanding, technical depth, methodology,
implementation and analysis. Current evidence is insufficient for a strong final
claim. The frozen SA learner now recovers six additional arrivals on its first
200 unseen scenarios, but safety differences remain uncertain. MA held-out and
SA official-protocol evaluations are running. Broader replication and the final
MA official-protocol evaluation remain pending. The number of experiments
alone is not evidence of innovation or a high grade.

## Work needed after the sprint decision

Both feasibility candidates are now frozen. If the team continues, finish their
unseen and official evaluations without further selection on those outcomes,
quantify training-seed variation, and make the learned contribution explicit. Report the entire metric
vector and per-aircraft/tail failures; lower average conflict obtained through
missed arrivals is not an adequate success claim. Retain unfavorable ablations.
For responsible-AI analysis, explain the limits of the finite conflict forecast
and inspect whether some aircraft bear disproportionate delay or risk. Initial descriptive
tail and simulation-slot analysis is recorded in ROBUSTNESS.md; it does not
establish causal fairness or real-world operational performance.

Prepare separate deliverables for each destination: the course allows a ten-page
report and requires its proposal/presentation; the competition asks for a 3-4 page
report, a five-scenario video and shareable code/model. Competition results use
its original 1000-scenario seed-42 harness. The local twenty-/200-scenario tables
and development GIFs are not a completed competition submission.

The course option remains conditional on the team's decision, as requested. A
well-supported negative result can still provide academic insight, but it does not
satisfy the team's separate ambition to build a controller that scores very well.
