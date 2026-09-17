# Why test a learned correction around fixed route guidance?

[Residual Policy Learning](https://arxiv.org/abs/1812.06298), Silver et al. (2018),
studies learning corrections to imperfect, potentially non-differentiable controllers
using model-free RL. The reported experiments concern robotic manipulation, not
this competition. [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201),
Johannink et al. (2018), similarly combines conventional control with learned
corrections and reports a physical block-assembly task. Primary abstracts checked
16 September 2026.

Our experimental inference is that a fixed obstacle-route reference may reduce the
navigation burden and leave a more tractable traffic-avoidance task for PPO. That
hypothesis is not evidence of ATC performance. Our wrapped/clipped bearing-offset
mapping is not an exact reproduction of either paper's controller architecture.

The registered experiment freezes route guidance and turns action filtering off.
Training adjusts aircraft heading offsets and speed actions using RL returns.
Evaluate the exact zero-experience policy with the same support flags; only its
paired difference from the trained policy measures learning. Separate initial
controls isolate guidance and filtering, and future trained ablations must retain
multiple training seeds and unseen scenarios. A high score from the fixed prior
alone does not satisfy the project goal.