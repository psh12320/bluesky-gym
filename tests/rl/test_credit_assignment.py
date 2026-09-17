import sys
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from atc_rl.buffer import ActiveRolloutBuffer
from atc_rl.policy import AircraftPolicy
from tests.rl.test_exploration import ConstantObservation

def fill(rewards, starts, values, gamma, lam):
    env=ConstantObservation()
    b=ActiveRolloutBuffer(len(rewards),env.observation_space,env.action_space,
                         device="cpu",gamma=gamma,gae_lambda=lam,n_envs=1)
    for r,start,value in zip(rewards,starts,values):
        b.pending_valid=np.ones(1,dtype=bool)
        b.add({k:np.zeros((1,*v.shape),dtype=np.float32) for k,v in env.observation_space.spaces.items()},
              np.zeros((1,2)),np.array([r]),np.array([start]),torch.tensor([value]),torch.zeros(1))
    return b

@pytest.mark.parametrize("lam",[.95,.99,1.])
def test_delayed_penalty_has_expected_credit_weight(lam):
    gamma=.996508469331006
    b=fill([0.]*36+[-1.],[True]+[False]*36,[0.]*37,gamma,lam)
    b.compute_returns_and_advantage(torch.zeros(1),np.ones(1,dtype=bool))
    expected=-np.power(gamma*lam,np.arange(36,-1,-1))
    assert b.advantages[:,0]==pytest.approx(expected,rel=2e-6,abs=1e-6)

@pytest.mark.parametrize("lam",[.95,.99,1.])
def test_credit_stops_at_terminal_and_bootstraps_at_rollout_cut(lam):
    b=fill([1.,2.,3.],[True,True,False],[.1,.2,.3],.9,lam)
    b.compute_returns_and_advantage(torch.tensor([4.]),np.zeros(1,dtype=bool))
    assert b.returns[0,0]==pytest.approx(1.)
    assert b.advantages[2,0]==pytest.approx(3+.9*4-.3)
    assert b.advantages[1,0]==pytest.approx(2+.9*.3-.2+.9*lam*(3+.9*4-.3))

@pytest.mark.parametrize("centralized",[False,True])
def test_lambda_survives_model_reload_and_buffer_reconstruction(tmp_path,centralized):
    torch.set_num_threads(1)
    model=PPO(AircraftPolicy,ConstantObservation(),seed=24,n_steps=4,batch_size=4,n_epochs=1,
              gamma=.996508469331006,gae_lambda=.99,device="cpu",
              policy_kwargs={"centralized":centralized,"actor_width":8,"critic_width":8})
    model.learn(8)
    path=tmp_path/"model.zip";model.save(path)
    restored=PPO.load(path,device="cpu")
    assert restored.gae_lambda==restored.rollout_buffer.gae_lambda==.99
    assert restored.gamma==restored.rollout_buffer.gamma==.996508469331006
    assert restored.policy.centralized==centralized

@pytest.mark.parametrize("value",["nan","inf","-0.01","1.01"])
def test_invalid_lambda_rejected_before_creating_run(tmp_path,value):
    path=tmp_path/"rejected"
    result=subprocess.run([sys.executable,"-m","atc_rl.train","--gae-lambda",value,"--run-dir",str(path)],
                          capture_output=True,text=True,cwd=Path(__file__).resolve().parents[2])
    assert result.returncode!=0
    assert "GAE lambda must be finite and in [0, 1]" in result.stderr
    assert not path.exists()
