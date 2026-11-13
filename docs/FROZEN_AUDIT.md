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
Gradient norms do not measure direction usefulness. There are no particle
correction steps, learned-generator updates, FID, or semantic-quality claims.

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
node. The launcher requests 80G host memory and one H100 using the same queue
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
