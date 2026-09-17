# Technical working draft

The four-page PDF at output/pdf/feasibility-working-draft.pdf summarizes completed
seed-2026 development evidence. It is an inspectable working draft, not a course or
competition submission. Its MA table uses ten-second control; the five-second MA
expansion is separate. No held-out outcomes were read to create this version.

The source tables are checked against complete CSV records, and all four rendered
pages have been inspected. The main text and tables use 12-point type. Small
nonzero safety values are preserved to four decimal places. The report includes
negative ablations, paired uncertainty, safety tails and limitations of the
classical comparison. It does not claim a competition ranking or operational
safety. Individual contributions and the course's tool-use declaration still need
accurate team review.

The report builders, evidence snapshot, figure and checks are in
runs/report-draft-v1/. Rebuild the figure with make_figure.py, then the PDF with
build_report.py. Python dependencies are NumPy, Matplotlib, ReportLab and pypdf;
the recorded environments describe the actual versions used. The six-entry
references.ris file can be imported into a reference manager. Import and final
citation-manager formatting have not yet been performed.

This version's primary evidence is fixed to the completed development runs listed
in evidence.json. A later held-out or official report should use a new version and
preserve this draft. It must not relabel these development results as official
scores. The selected source/model hashes and exact callback update counts are
recorded in qa.json and the individual evaluation records.

## Later evidence

The preserved PDF above predates the completed held-out comparisons, training-seed
replications and the first full scoring result. Current tables are in
HELDOUT_RESULTS.md, REPLICATIONS.md and SCORING_RESULTS.md. Original-format SA
full scoring has 98.5% arrivals and 70.4% clean completion; corrected SA/MA full
scoring remain pending. All declared training-seed replications are complete. Final report input checks
are prepared under runs/report-final-v1 and refuse a final evidence snapshot
until those declared evaluations are complete. No replacement final PDF has been
written yet. The four-page report builder is prepared, but has not been run against incomplete inputs or rendered for visual review.


A further attributed development experiment is now declared in
INTERVENTION_REWARD.md. It tests training feedback for command-filter corrections
with matched training and unchanged physical scoring. It has no completed
performance result yet; the eventual report must record its actual disposition.


Report polishing is paused under the revised RL priorities. RL_PLAN.md defines the
remaining sprint. The prepared report builder must not be treated as the active
development objective; PPO/MAPPO learning evidence comes first.
