# First-stage objectives

An episode has generated latent queries $g_1,\ldots,g_Q$ and real references
$r_1,\ldots,r_R$, all from one class. Both encoders see the same states and the
same selected feature positions. Each branch supplies a tensor [position, sample,
channel]. Means/stds over an entire feature map are separate branches.

For encoder $E$, position $p$, and one branch, let

$$d^+_{E,pij}=\|E(g_i)_p-E(r_j)_p\|_2,\qquad
d^-_{E,pik}=\|E(g_i)_p-E(g_k)_p\|_2\quad(k\ne i).$$

Set $s_{E,p}=\operatorname{mean}_{ij}d^+_{E,pij}$ and
$\bar d^{\pm}_{E,p}=d^{\pm}_{E,p}/s_{E,p}$.
Distances and losses are computed in float32. For encoder pretraining the student
scale is differentiated; the teacher features and scales are detached. Zero
distance-scale episodes are rejected instead of silently producing uniform
targets.

At each temperature $\tau$, define row-normalized affinities
$A^\pm_E(\tau)=\operatorname{softmax}(-\bar d^\pm_E/\tau)$.
The geometry arm minimizes

$$L_K=\operatorname{mean}_{\ell,p,\tau}
\frac{\mathrm{KL}(A^+_T\|A^+_S)+\mathrm{KL}(A^-_T\|A^-_S)}2
+0.1\operatorname{mean}_{\ell,p}
\frac{\mathrm{Huber}(\bar d^+_S,\bar d^+_T)+
\mathrm{Huber}(\bar d^-_S,\bar d^-_T)}2.$$

KL additionally averages query rows; Huber averages all valid pairs. Every
branch receives equal outer weight. Negative self-pairs are removed before
softmax. No equality to the original normalized stopped-field objective is
claimed.

The feature-copy arm uses exactly the same retained adapters as the geometry arm.
For each branch, divide both student and teacher values by the same detached
teacher RMS $\sqrt{\operatorname{mean}T^2+10^{-6}}$, then minimize mean squared
difference. Independent normalization of student/teacher values is deliberately
avoided because that would discard amplitude information.

The reconstruction control masks contiguous 4×4 latent patches with probability
0.5, zeros masked inputs, and minimizes mean squared error over masked coordinates
only. If a sample happens to have no masked patch, one patch is masked. Its decoder
is not part of the eventual feature encoder.

Matching distances/affinities on cached samples does not determine an encoder's
input Jacobian. Exact generator-facing gradient checks and matched generator
continuations are required after this first-stage test. Selecting a geometry
encoder by geometry fidelity alone cannot establish good FID or useful semantics.
