# Relative-position conditioning experiment

The default remains traffic_position_scale=1.0. The optional --traffic-position-scale 20 multiplies only the existing body-frame x_r/y_r coordinates by twenty. Their original normalization is one million metres; this choice expresses them in units of 50 km. No clipping, prediction, new information or controller action is added. All native scoring records, dynamics, scenario generation, population and action timing remain fixed.

Both the actor's local observation and the centralized critic's corresponding joint observations receive the same transform. This tests input conditioning throughout the learner; it cannot isolate an actor-only effect. Input dimensionality and parameter counts stay unchanged. With a neutral Gaussian actor and identical initial tensors, deterministic means and initial action distributions remain identical; initial critic values can differ because their inputs differ.

The option is reconstructed from checkpoint metadata in both the vector evaluator and native single-/multi-aircraft deployment. Comparisons reject mismatched scales. Existing Gaussian/gSDE choices remain separate experimental factors. Previously frozen source bundles and cluster studies retain their original settings.

The saved-data audit in runs/traffic-input-sensitivity-v1/results.json inspected all completed models from two CPA feature/mask seed pairs and both Gaussian-width arms. On 371 reference observations, relative positions accounted for about 0.1–0.36% of squared standardized heading gradients, while other groups had larger values. This is a diagnostic on one reused trajectory, not causal feature importance, proof of a defect, or evidence that rescaling improves learning.

The proposed controlled test uses Gaussian PPO, identical architecture, initial tensors, action-noise scale, budgets, route/static support and training seeds, comparing scale 1 with scale 20. Evaluate each policy against its own initial policy and the fixed classical benchmark; repeat across seeds if a pilot warrants it. Do not combine this comparison with a change to exploration timing.
