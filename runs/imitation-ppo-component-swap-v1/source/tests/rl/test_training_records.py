import csv
import os
from pathlib import Path
import subprocess
import sys
import pytest
from atc.metrics import METRICS
from atc_rl.training_records import TrainingRecords


def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def test_completed_records_survive_abrupt_process_exit_without_final_summary(tmp_path):
    code="""
import os,sys
from atc.metrics import METRICS
from atc_rl.training_records import TrainingRecords
from pathlib import Path
p=Path(sys.argv[1]);writer=TrainingRecords(p,.01)
aircraft=[{k:0. for k in METRICS}];aircraft[0]['total_reward']=-10.
writer.append(aircraft,[-.08]);writer.append(aircraft,[-.08])
aircraft.append({k:1. for k in METRICS});writer.append(aircraft,[-.08,.01])
os._exit(0)
"""
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True)
    aircraft=read(tmp_path/'training-aircraft.csv');returns=read(tmp_path/'training-returns.csv')
    assert len(aircraft)==len(returns)==2
    assert [r['completion_index'] for r in returns]==['0','1']
    assert float(returns[0]['native_return'])==-10.
    assert float(returns[0]['unscaled_learning_return'])==-8.
    assert float(returns[0]['shaping_return'])==2.
    assert float(returns[1]['shaping_return'])==0.
    assert not (tmp_path/'training_summary.json').exists()


def test_record_writer_rejects_misaligned_returns_and_preserves_existing_files(tmp_path):
    with TrainingRecords(tmp_path,1.) as writer:
        with pytest.raises(ValueError,match='aligned'):writer.append([{k:0. for k in METRICS}],[])
    before=(tmp_path/'training-aircraft.csv').read_bytes()
    with pytest.raises(ValueError,match='already exist'):TrainingRecords(tmp_path,1.)
    assert (tmp_path/'training-aircraft.csv').read_bytes()==before
