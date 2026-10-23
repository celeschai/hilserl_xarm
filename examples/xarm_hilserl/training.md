# Training (SAC + DQN gripper)

Learner: `lerobot.rl.learner` with `configs/train.json`. Code: `src/lerobot/rl/algorithms/sac/sac_algorithm.py`, `src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`.

## Data

**Transition** $(s, a, r, r_g, s', d)$

| Symbol | Meaning |
|---|---|
| $s = (I_{\text{wrist}}, I_{\text{side}}, I_{\text{front}}, x)$ | 3 images $\in \mathbb{R}^{3\times128\times128}$ (ImageNet-normalized) + state $x \in [-1,1]^{19}$ |
| $x$ | relative pose (6), EE-frame twist (6), wrench (6), gripper (1) |
| $a = (a_c, a_g)$ | $a_c \in [-1,1]^6$ EE-frame delta pose, $a_g \in \{0,1,2\}$ = close / stay / open |
| $r$ | $\mathbb{1}[\text{classifier}(s') > 0.5]$ |
| $r_g$ | $-0.02$ if $a_g$ toggles the gripper, else $0$ |
| $d$ | $1$ on success (episode ends), else $0$ |

**Batch** $B$: 256 transitions, $\frac{1}{2}$ online buffer $\mathcal{D}_{\text{on}}$, $\frac{1}{2}$ offline buffer $\mathcal{D}_{\text{off}}$ (demos + interventions).

## Networks

**Encoder** (shared) $\phi: s \mapsto \mathbb{R}^{1024}$

$$z_k = \tanh\big(\mathrm{LN}(W_k\,\mathrm{SpatialEmb}_k(\mathrm{ResNet10}(I_k)))\big) \in \mathbb{R}^{256}, \qquad z_x = \tanh(\mathrm{LN}(W_x x)) \in \mathbb{R}^{256}$$

$$\phi(s) = [z_{\text{wrist}}, z_{\text{side}}, z_{\text{front}}, z_x]$$

ResNet10 is frozen. $W_k$, $\mathrm{SpatialEmb}_k$, $W_x$ are trained by the critic loss only (detached in the actor).

| Network | Input | Output |
|---|---|---|
| Actor $\pi_\theta$ | $\phi(s)$ | $\mu, \sigma \in \mathbb{R}^6$; $a_c = \tanh(u),\ u \sim \mathcal{N}(\mu, \sigma^2)$ |
| Critics $Q_{1,2}$ | $[\phi(s), a_c]$ | $\mathbb{R}$ |
| Gripper critic $Q^g$ | $\phi(s)$ | $\mathbb{R}^3$ (one value per $a_g$) |
| Temperature $\alpha$ | — | $\alpha = e^{\log\alpha}$, $\alpha_0 = 0.01$ |

$\bar Q_{1,2}, \bar Q^g$ are target copies.

$$\log\pi(a_c|s) = \log\mathcal{N}(u;\mu,\sigma) - \textstyle\sum_{i=1}^{6}\log(1-\tanh^2 u_i)$$

## Losses (per update, $\gamma = 0.97$)

**Critic**, with $a' \sim \pi(\cdot|s')$:

$$y = r + \gamma(1-d)\Big[\min_{i}\bar Q_i(s',a') - \alpha\log\pi(a'|s')\Big]$$

$$\mathcal{L}_Q = \sum_{i=1}^{2}\mathbb{E}_B\big[(Q_i(s,a_c) - y)^2\big]$$

**Gripper critic** (Double DQN):

$$y^g = r + r_g + \gamma(1-d)\,\bar Q^g(s')_{g^*}, \qquad g^* = \arg\max_g Q^g(s')_g$$

$$\mathcal{L}_g = \mathbb{E}_B\big[(Q^g(s)_{a_g} - y^g)^2\big]$$

**Actor**, with $\tilde a \sim \pi(\cdot|s)$:

$$\mathcal{L}_\pi = \mathbb{E}_B\big[\alpha\log\pi(\tilde a|s) - \min_i Q_i(s,\tilde a)\big]$$

**Temperature**, target entropy $\bar{\mathcal{H}} = -7/2$:

$$\mathcal{L}_\alpha = \mathbb{E}_B\big[-\alpha\,(\log\pi(\tilde a|s) + \bar{\mathcal{H}})\big]$$

**Targets** ($\tau = 0.005$):

$$\bar\theta \leftarrow \tau\,\theta + (1-\tau)\,\bar\theta$$

Each loss: Adam, lr $3\times10^{-4}$, grad-norm clip 40. Order per step: $\mathcal{L}_Q \to \mathcal{L}_g \to \mathcal{L}_\pi \to \mathcal{L}_\alpha \to$ targets.

## Output

**Policy** (sent to the actor every 2 steps):

$$a = \big[\tanh(u),\ \arg\max_g Q^g(s)_g\big], \qquad u \sim \mathcal{N}(\mu(s), \sigma(s)^2)$$

The env scales $a_c$ by 0.01 m (translation) and 0.06 rad (rotation) per step.
