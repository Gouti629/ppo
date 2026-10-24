# PPO from Scratch: LunarLander-v3 and Swimmer-v5

A from-scratch PyTorch implementation of Proximal Policy Optimization (Schulman et al., 2017), applied to a discrete control task (LunarLander-v3, Gymnasium) and a continuous control task (Swimmer-v5, MuJoCo). The goal was to move from value-based methods to policy-gradient and actor-critic methods hands-on, and to understand empirically how PPO's core hyperparameters interact on both kinds of action spaces.

## Implementation

- Single-actor PPO with the clipped surrogate objective (eps = 0.2), following the original paper
- Generalized Advantage Estimation (gamma = 0.999, lambda = 0.95), with advantage normalization
- Entropy bonus annealed linearly over training (0.01 -> 0.0005) to decay exploration as the policy matures
- Gradient clipping (max norm 0.5) and a linearly decaying Adam learning rate (initial 1e-3)
- Policy and value networks: two hidden layers of 256 units; ReLU activations for LunarLander, Tanh for Swimmer (bounded actions in [-1, 1])
- Continuous actions modeled with a Gaussian policy and a learnable log-standard-deviation parameter
- One shared training loop handles both discrete and continuous action spaces
- Policy evaluated every 5% of total training steps by averaging returns over 200 evaluation episodes

## Experiments

I swept the three hyperparameters I expected to matter most, on both environments:

- K_epochs (update epochs per batch): 8 to 40
- T_horizon (steps collected between updates): 1024, 2048, 3072
- batch_size: 32, 64

### LunarLander-v3 (discrete)

- Best configuration: T_horizon = 2048, batch_size = 32, K_epochs = 32, with average evaluation reward crossing 150
- The effect of K_epochs was non-monotonic: 8 epochs reached ~150, 24 epochs performed the worst (~50), and 32 epochs performed best. There was no clean trend between epochs and final reward
- Both 1024 and 3072 horizons underperformed 2048, suggesting a sweet spot for how much on-policy data this task needs before each update
- Solve threshold was not reached: Gymnasium considers 200 "solved." A single run touched 199, but a repeat run with identical hyperparameters landed just above 50, so run-to-run stochasticity remains the open problem

### Swimmer-v5 (continuous, run on Colab GPU)

- T_horizon = 3072 produced the best results; shorter horizons performed poorly
- Best runs crossed an average reward of 40. The original paper reports over 100 on Swimmer, but trained for roughly 10x more steps
- Compute limits forced single runs per configuration (repeat attempts died with Colab disconnects), so these results carry high variance and are directional, not conclusive

## What I took away

- The three swept hyperparameters do not act independently. Ratios like T_horizon / batch_size or products like K_epochs * batch_size look more meaningful than any single knob, since they describe how many gradient steps you take per unit of fresh data. Testing that hypothesis is future work
- High variance across identical runs was the biggest obstacle to drawing conclusions; multiple seeds per configuration are a prerequisite for any real claim
- Following the paper alone was not enough. The fixes that made training work (advantage normalization, gradient clipping, entropy annealing) came from implementation-detail writeups, especially Costa Huang's "The 32 Implementation Details of PPO"

## What I would do differently

- Save reward curves to disk after every run instead of keeping them in the notebook, so a crashed runtime does not wipe hours of experiments
- Plot all sweep configurations on one chart instead of one chart per run
- Use proper weight initialization (e.g., Glorot) and test deeper networks
- Fix a seed budget up front and run every configuration across multiple seeds

## Repo layout

- PPO.ipynb - full implementation, training runs, and evaluation plots
- Project Report.pdf - writeup covering the algorithm, design decisions, experiment results, and lessons learned

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Costa Huang, "The 32 Implementation Details of PPO" (costa.sh)
- OpenAI Spinning Up: PPO