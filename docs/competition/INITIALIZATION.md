# PPO initialization ablation

Status: the neutral-mean/std-0.05 direct-heading pilot completed and failed its registered advancement screen.

The scenario generator points each aircraft's initial heading toward its goal
(core/scenario.py). The current PPO Gaussian starts at log standard deviation
-0.5: approximately 0.6065 normalized units, or 27.29 degrees of requested heading
change before command clipping. This describes requested commands, not actual
A320 turn rates. The initial mean also has small nonzero neural-network outputs.
Repeated relative-heading commands can therefore make initial rollouts quite
different from the neutral straight-flight behavior.

Two independent controls are now available in the development trainer:

- --initial-action-std changes only the initial Gaussian standard deviation.
  The default remains 0.6065306597126334; a candidate smaller value is 0.05.
- --neutral-action-mean zeros only a fresh actor's output weight and bias.
  Hidden actor weights, critic weights and the learner RNG remain unchanged.
  This is an initialization: no fixed controller replaces subsequent actions.

A future controlled experiment should use the four combinations of original or
smaller standard deviation and original or neutral mean. Keep reward, observation,
action interval, worker count, seed, budget and optimizer settings fixed within
this comparison. Do not attribute differences from reward scaling or route/filter
support to initialization. Measure every configuration against its own untrained
actor and the fixed classical benchmark on paired development scenarios. A high
neutral-policy arrival rate alone is not a learned contribution: safety and the
incremental benefit of PPO updates are essential.

The unit checks verify exact preservation of unrelated weights and RNG, exactly
zero initial means, expected standard deviation, subsequent PPO updates, rejection
of attempts to overwrite trained actors, and identical mean/value weights when
only initial exploration changes. Full simulator performance remains unmeasured.


### First measured control

The neutral policy with standard deviation 0.05 has no optimizer updates or
simulator training. On 20 development worlds it achieved 200/200 arrivals,
19.5% clean completion, 73.44 seconds intrusion time and 126.975 seconds in
restricted areas. All thirteen parameter tensors exactly match the associated
training run's initial checkpoint. That run completed 101,525 live transitions,
897 optimizer updates and 181 epochs; its performance evaluation is pending.
The two single-factor initialization cells are not yet trained. Do not describe
the full four-cell initialization ablation as complete.


### Completed development evaluation

After training, arrival was 4% (8/200) and clean completion 2%, versus 100% and 19.5% for its exact initial policy. Mean intrusion/restricted/outside times were 132.53 / 149.715 / 1937.42 seconds. See runs/ppo-initialization-v1/comparison-neutral-dev20 and advancement-screen.json. The other two initialization cells remain untrained; effort has moved to the goal-offset representation.
