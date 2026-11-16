# First frozen-encoder diagnostic

This is a deliberately small real-checkpoint smoke/diagnostic before student
pretraining. It needs no generator checkpoints. It uses four query and eight
independent real reference images from one ImageNet TRAIN class, selected by a
fixed seed. Repeat across classes/seeds later before drawing conclusions.

The VAE samples scaled latents once using seed42. Both encoder runs receive the
same states. Cases: unchanged queries, additive latent noise (std0.15), latent
3x3 average blur, and a shared seeded permutation of 4x4 latent patches.
These latent perturbations are probes, not established perceptual rankings.
Inspect saved decoded PNGs to understand what they actually change. The clean
control also differs from its independent reference batch: finite-sample MMD
need not be zero.

MAE: five stage GLOBAL MEANS, not the full original multiscale loss recipe.
Inception: ImageNet-supervised torchvision Inception-v3 pooled features,
frozen/eval mode, decoded RGB, direct bilinear resize299 and ImageNet channel
normalization. This is not torch-fidelity's FID Inception. No encoder is trained.

Both use a conventional biased Laplace-kernel MMD2 diagnostic, NOT the stopped
normalized Riesz-field surrogate. Each branch bandwidth is its real-reference
median pair distance; it stays fixed across cases and backward. Branch losses
are averaged. Raw MMD values across different encoders are not comparable.
Compare within-encoder changes versus the clean control; the small sample count
means these measurements are exploratory, not statistical evidence of ranking.

Timing covers query forward, this MMD calculation, and backward to query LATENTS
with references cached; Inception includes VAE decoding. Warmup and checkpoint
loading/reference extraction are outside timing. It retains each query's forward
graph and concatenates feature microbatches; microbatch1 does not eliminate all
retained activations. The default query count4 limits initial memory demand.
Do not infer full original-recipe speedup from this limited branch comparison.
Gradient norms do not measure direction usefulness. The default run has no
correction steps; the optional extension below probes directions. There are no
learned-generator updates, FID, or semantic-quality claims.

## Setup on Beaker

Finish the VAE download first. Clone this repository on Beaker if it only exists
on Scafell. Activate a separate copy of the working image-model environment:

    conda create -n feature-audit --clone mmd_flow
    conda activate feature-audit
    python -m pip install -r requirements-audit.txt

Download the Inception weights on the login node, before the GPU job:

    python -c 'from torchvision.models import inception_v3, Inception_V3_Weights; inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)'

The MAE artifact and VAE folder must be present at the paths in the launcher.
MAE loading has a substantial host-RAM peak; use an allocated job, not the login
node. The launcher requests 80G host memory and one H100/A100/V100 using the same queue
conventions as the user's existing cluster scripts. Activate feature-audit and
run qsub from this repository so -cwd/-V preserve the source/env.

    export FEATURE_AUDIT_ENCODER=mae
    export FEATURE_AUDIT_OUTPUT=/SAN/intelsys/imagenet_mmd_flow/compact_feature_audits/mae_seed42
    qsub scripts/audit_sge_1gpu.sh

After it finishes, submit the corresponding Inception job:

    export FEATURE_AUDIT_ENCODER=inception
    export FEATURE_AUDIT_OUTPUT=/SAN/intelsys/imagenet_mmd_flow/compact_feature_audits/inception_seed42
    qsub scripts/audit_sge_1gpu.sh

Each output must be a fresh directory. Both summaries record image hashes, seed,
normalization, reference scales, pretrained MAE provenance when applicable, GPU
name, and measurement scope. Confirm selected image hashes match between arms.
A fair repeated study should additionally pin/record torchvision/VAE weight
hashes and software versions, audit more classes/seeds, vary kernel bandwidths,
and test predictions in actual generator training. No real GPU audit was run in
the development environment; helper tests only verify the diagnostic algebra.

The updated launcher accepts h100, a100, or v100 using Grid Engine's quoted OR
expression. Both encoder paths use FP32 in this audit; the MAE metadata precision
setting is overridden for this diagnostic and logged. This avoids relying on
native BF16 support on older GPUs. No real run has yet established memory fit on
each GPU type. Keep comparisons on the same GPU model when reporting relative
timings; do not compare an Inception timing on V100 with an MAE timing on H100.
The 80G resource is HOST RAM for checkpoint construction/conversion, not VRAM.
Use smaller GPUs for the Inception path after measuring its requirements.
Already queued jobs retain their spooled resource request; pulling a changed
launcher does not broaden an existing job's GPU eligibility.

## Optional short correction probe

Set `FEATURE_AUDIT_CORRECTION_STEPS=10` and `FEATURE_AUDIT_STEP_RMS=0.01`
before submission, with a fresh output directory for each encoder. This adds
ten deterministic descent steps on the same fixed-reference MMD, for every
case including the unchanged clean control. Bandwidths remain fixed. Each
particle's gradient is normalized by its own RMS, giving both encoders the
same per-coordinate latent update budget. This is a diagnostic choice, not a
tuned sampling algorithm or the original Riesz training surrogate. A zero
gradient gives no movement. No line search or ground-truth-guided step selection
is used. The chosen step size may be inappropriate; inspect the whole trace.

`correction_trace.json` records MMD, latent displacement, paired latent MSE,
and cosine of the descent direction toward the original clean latent at every
step. Null cosine means the original clean state has already been reached.
At steps 0, 1, 5 and the final step, decoded PNGs and unclipped decoded RGB MSE
are saved. Displayed PNGs are clipped to [0,1]. Decoding is outside the timing
measurement. MAE is released before reloading the decoder for these previews.

In the default MMD mode the paired clean queries are only used for reporting,
never by the descent objective. References are different same-class images. Distribution matching
therefore need not recover a particular original image; exact recovery is an
auxiliary diagnostic, not a requirement for a valid distributional gradient.
Clean-control drift can reveal finite-reference overfitting. Reduced pixel MSE
can also reward smoothing, so inspect the grids. This tiny one-class probe cannot
rank encoders generally. Replicate across seeds/classes and test actual generator
updates before making an encoder-quality claim.

## Paired clean-target sanity control

Set `FEATURE_AUDIT_CORRECTION_OBJECTIVE=paired_feature` to use each query's own
clean features as its frozen target. Unlike MMD mode, this is explicitly paired
feature regression: for each branch, sum squared feature-coordinate errors,
divide by the fixed independent-reference median distance squared, then average
samples and branches. The unchanged MMD against independent references is still
reported as a diagnostic; it is not the optimized objective in this mode.

The same images, perturbations, ten steps, and per-particle latent RMS budget
are used. A paired error <=1e-10 suppresses that particle's update to avoid
amplifying roundoff at its clean optimum. This threshold is recorded. Targets
receive no gradients. No pixel/latent reconstruction loss is optimized.

Inspect `objective_value`, paired latent and decoded RGB errors, directional
cosines, and saved grids. Clean controls should stay still. Feature regression
can still change texture without recovering the original: globally pooled
features are many-to-one, and fixed-size steps may overshoot. A negative result
after ten steps is not proof that the encoder lacks useful gradients. A positive
result is not evidence that it trains a generator well. This is only an inversion
sanity control, separate from the population-matching experiment.

On Beaker, after activating the SAN feature-audit environment and pulling:

    bash scripts/submit_paired_control.sh

This submits one job for each encoder and prints job IDs/log paths. Existing
audit outputs are not changed. The actual GPU path must be checked on cluster;
local tests verify the paired loss, gradients, target detachment, and clean
zero-update control only.
