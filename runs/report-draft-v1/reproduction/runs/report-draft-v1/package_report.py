from pathlib import Path
import hashlib, json, zipfile
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'runs/report-draft-v1'
output=ROOT/'output/pdf/feasibility-working-draft-evidence.zip'
assert not output.exists()
files={
 'output/pdf/feasibility-working-draft.pdf',
 'docs/competition/REPORT.md','docs/competition/references.ris','docs/competition/ROBUSTNESS.md',
 'runs/report-draft-v1/build_report.py','runs/report-draft-v1/make_figure.py',
 'runs/report-draft-v1/evidence.json','runs/report-draft-v1/build.json','runs/report-draft-v1/qa.json',
 'runs/report-draft-v1/environments.json','runs/report-draft-v1/package_report.py',
 'runs/report-draft-v1/sa-distributions.png','runs/report-draft-v1/sa-distributions.svg',
 'atc/__init__.py','atc/compare.py','atc/metrics.py',
}
evidence=json.loads((folder/'evidence.json').read_text())
for item in evidence['evaluations'].values():
    files.update(item['prefix']+suffix for suffix in ('.csv','.json'))
files.update('runs/fast-reference-v1-ma25k-sa/validation-200'+suffix for suffix in ('.csv','.json'))
manifest={'purpose':'Reproduce the development-only working report; trained-policy evaluation requires the main repository and model files',
          'no_heldout_data':True,'model_weights_included':False,'source_sha256':{}}
with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(files):
        data=(ROOT/name).read_bytes()
        manifest['source_sha256'][name]=hashlib.sha256(data).hexdigest()
        archive.writestr(name,data)
    archive.writestr('manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in manifest['source_sha256'].items())
record={'file':str(output),'files':len(files),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'crc_and_source_hashes_verified':True,'no_heldout_data':True}
(folder/'package.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
