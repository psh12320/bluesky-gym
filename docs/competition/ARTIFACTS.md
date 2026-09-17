# Research artifacts

The `research-handoff-20260917` release accompanies `AGENT_HANDOFF.md`. Git contains source, tests and text experiment evidence. Release ZIP files contain local models, checkpoints, demonstrations, frozen source bundles and media. `replay-*.zip` separately preserves historical replay bytes; these are archival only. **Do not deserialize historical replay pickle files.**

The release contains fourteen ZIP files (4.69 GB total): one 0.93 GB model/data/source archive and thirteen optional historical replay archives. There are 7,313 text experiment records in Git and 1,376 binary files mapped to release archives. The index has 626 retained run/record groups.

The manifest in `artifacts/manifest.json` maps every included file to its exact path, size, SHA-256 and storage location. It also records excluded caches, dependencies, generated test fixtures, duplicate expanded source bundles and reference-only material. Local originals remain intact. Remote-only cluster artifacts are absent: receipts are not a substitute for downloading them.

Clone the competition branch:

```bash
git clone --branch AI4REAL-NET-Competition https://github.com/psh12320/bluesky-gym.git
cd bluesky-gym
```

Download ordinary model/data/source artifacts with GitHub CLI (or select the same assets from the release page):

```bash
gh release download research-handoff-20260917 --repo psh12320/bluesky-gym --pattern 'artifacts-*.zip' --dir downloads
```

For each downloaded ZIP, run the portable Python restoration script. Example:

```bash
python scripts/restore_research_artifacts.py downloads/artifacts-01.zip
```

The script checks archive and member checksums, refuses paths outside `runs/`, skips identical existing files and refuses to overwrite differing files. `--verify-only` checks the whole archive without extracting. It does not load any model/pickle object. Git attributes preserve evidence bytes across Windows/Linux checkouts.

Archival replay files are optional and unnecessary for the proposed next experiment. To preserve a complete binary copy, also download `replay-*.zip`; they can be verified/restored using the same script, without deserializing them.

Expanded `*-verify` source directories are generally omitted when their original source ZIP exists. Restore the ZIP, verify its manifest, and extract into a fresh directory if reproducing that exact experiment. Original protocols/runners contain machine-specific absolute paths. Adapt new copies for another machine; retain historical source/protocol files unchanged.

The experiment index inventories evidence, not successful experiments. Some historical state files are stale. Use completed summary/CSV/audit evidence and the current handoff for interpretation. Older report drafts remain historical.