"""Check Linux simulator results against the recorded Windows adapter fixture."""
from pathlib import Path
import argparse
import csv
import json
import math


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'tests/rl/fixtures/adapter-reference.json').read_text(encoding='utf-8'))
    scenarios=json.loads((args.directory/'scenarios.json').read_text(encoding='utf-8'))
    if scenarios!=expected['scenarios']:raise ValueError('Generated scenario fingerprints differ between platforms')
    with (args.directory/'aircraft.csv').open(newline='',encoding='utf-8') as stream:
        actual=list(csv.DictReader(stream))
    fields=expected['metric_fields']
    keyed=lambda rows:{(int(r['episode']),r['agent']):r for r in rows}
    reference=keyed(expected['aircraft']);observed=keyed(actual)
    if reference.keys()!=observed.keys():raise ValueError('Completed aircraft identities differ')
    for key in reference:
        for metric in fields:
            if not math.isclose(float(reference[key][metric]),float(observed[key][metric]),rel_tol=0,abs_tol=1e-6):
                raise ValueError(f'Cross-platform metric differs: {key} {metric}')
    summary=json.loads((args.directory/'summary.json').read_text(encoding='utf-8'))
    if not summary['padded_transitions'] or summary['aircraft_terminated']!=20:
        raise ValueError('Adapter did not exercise termination and padding')
    print('Two scenario fingerprints and all 180 aircraft metrics match the Windows reference.')


if __name__=='__main__':main()
