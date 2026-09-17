"""Training-only feedback for the unchanged joint command filter.

Action-adjustment reward penalties are established prior work; see
https://arxiv.org/html/2509.12833v2 (section 7.1). This experiment uses a
bounded heading/speed normalization, without assuming a formal safety guarantee.
"""
import math
from pettingzoo.utils import BaseParallelWrapper

REVISION = 1

def correction_cost(record):
    if not record['traffic_projection_intervened']:
        return 0.0
    nominal_heading = record['reference_heading_turn_deg'] + record['nominal_heading_adjustment_deg']
    executed_heading = record['commanded_heading_turn_deg']
    nominal_speed = record['nominal_speed_action']
    executed_speed = record['commanded_speed_action']
    values = (nominal_heading, executed_heading, nominal_speed, executed_speed)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('Correction feedback requires finite commands')
    if any(abs(value)>45+1e-5 for value in values[:2]) or any(abs(value)>1+1e-6 for value in values[2:]):
        raise ValueError('Correction feedback assumes turns within 45 degrees and normalized speed')
    cost = .5 * (((executed_heading-nominal_heading)/90)**2 + ((executed_speed-nominal_speed)/2)**2)
    if not 0 <= cost <= 1+1e-6:
        raise ValueError('Invalid normalized correction cost')
    return min(cost,1.0)

class InterventionReward(BaseParallelWrapper):
    """Change returned training rewards while preserving observations and scoring info."""
    def __init__(self, env, coefficient):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Penalty coefficient must be finite and nonnegative')
        super().__init__(env)
        self.coefficient = float(coefficient)
        self.statistics = dict(steps=0, commands=0, interventions=0, correction_cost=0.0,
                               penalty=0.0, original_reward=0.0, training_reward=0.0)

    def step(self, actions):
        world=self.env.unwrapped
        previous=dict(world.last_projection)
        observations,rewards,terminated,truncated,infos=self.env.step(actions)
        if set(rewards)!=set(actions):
            raise ValueError('Feedback must cover precisely the live aircraft that acted')
        shaped={}
        for agent,original in rewards.items():
            record=world.last_projection[agent]
            if previous.get(agent) is record:
                raise ValueError('Refusing stale command-filter diagnostics')
            cost=correction_cost(record)
            penalty=self.coefficient*cost
            shaped[agent]=original-penalty if penalty else original
            self.statistics['commands']+=1
            self.statistics['interventions']+=int(record['traffic_projection_intervened'])
            self.statistics['correction_cost']+=cost
            self.statistics['penalty']+=penalty
            self.statistics['original_reward']+=float(original)
            self.statistics['training_reward']+=float(shaped[agent])
        self.statistics['steps']+=1
        return observations,shaped,terminated,truncated,infos
