# Compact particle features

A first-stage experiment: can a small latent encoder learn enough of an existing
MAE teacher's comparison geometry to become a useful, cheaper loss encoder?

**Implemented:** portable latent packets, fixed teacher-feature episode caches,
three student-training objectives, validation/audit geometry diagnostics,
student timing, restart support, and one-GPU/four-GPU launchers.

**Not yet implemented:** inserting the student into the full ImageNet generator
training job, exact stopped-Riesz-field input-gradient audits, or FID evaluation.
Passing this pilot alone does not establish an MAE replacement. No trained
weights or ImageNet data are included.

## Start on Scafell

Use Python 3.11 and a working CUDA PyTorch environment. Scafell's existing
`dmmd` environment has a compatible PyTorch; copying it avoids changing other
experiments:

```bash
git clone https://github.com/seulkang0518/compact-particle-features.git
cd compact-particle-features
conda create -n compact-features --clone dmmd
conda activate compact-features
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
bash scripts/smoke.sh cuda
```

The smoke test creates a tiny **synthetic** teacher cache and runs all three
training/evaluation paths. It needs neither MAE weights nor ImageNet. Use
`bash scripts/smoke.sh cpu` on a CPU. To repeat it, supply a fresh output directory:
`bash scripts/smoke.sh cuda runs/synthetic_smoke_2`.

## What the three arms learn

All arms use a four-channel latent CNN with stem width 64, four residual stages
of widths 64/128/256/512, GroupNorm, SiLU, and two blocks per stage.

| CLI arm | Training signal | Retained loss encoder |
|---|---|---|
| `reconstruction` | Reconstruct masked latent patches | Native compact encoder; decoder discarded |
| `feature` | Copy teacher features using normalized MSE | Compact encoder **plus channel adapters** |
| `geometry` | Match teacher attraction/repulsion affinities and normalized distances | Exactly the same encoder/adapters as `feature` |

The teacher is frozen. The small real-data pilot uses 8 generated queries and
16 real references per same-class episode; the larger proposed pilot uses
16 queries and 32 references when all held-out splits have enough data.
Geometry matching uses generated-to-real and
generated-to-other-generated distances; generated self-pairs are excluded.
Temperatures are 0.2, 0.05, 0.02. This is teacher-guided compression, **not**
unrestricted maximization of MMD and not an exact implementation of the existing
stopped Riesz movement loss. See [the objectives](docs/OBJECTIVES.md).

Five map branches (`conv1`, `layer1`–`layer4`) expose selected spatial locations
plus full-map means and standard deviations. Intermediate-block and local-patch
branches from the original full MAE recipe are omitted in this pilot. Both
teacher-guided arms keep identical output heads so their objective comparison
is meaningful. The reconstruction arm has different output channel widths.

## Real ImageNet pilot

1. On the cluster, prepare a packet from **already scaled** VAE latents and saved
   generator outputs. Split original real-image IDs and noise identities before
   constructing episodes. Hold out an entire generator checkpoint for audit.
2. Cache the frozen MAE's sampled features once. This needs the existing
   `riesz_flow_sophia` source checkout and its local `mae_latent_640` artifact.
3. Copy the packet/cache to Scafell if desired. Student training reads the cache;
   it does not load the MAE.
4. Train matched student arms, select using validation only, then inspect the
   held-out audit once settings are fixed.

Exact array formats and commands are in [DATA.md](docs/DATA.md). A real generator
snapshot exporter is not bundled: this release accepts generated-latent NPZ
files with checkpoint, class, CFG, and noise provenance. Do not replace these
with newly corrupted real images and call it the same experiment.

```bash
# After constructing data/teacher_cache (see docs/DATA.md):
bash scripts/train_scafell.sh data/teacher_cache feature runs/feature_seed42
bash scripts/train_scafell.sh data/teacher_cache geometry runs/geometry_seed42
bash scripts/train_scafell.sh data/teacher_cache reconstruction runs/reconstruction_seed42

python evaluate.py --cache data/teacher_cache \
  --checkpoint runs/geometry_seed42/latest.pt --split val --device cuda \
  --output runs/geometry_seed42/geometry_val.json
```

`latest.pt` is the fixed-budget endpoint; `best.pt` uses that arm's own validation
objective. Loss numbers from different arms are not directly comparable.
For the initial controlled comparison, evaluate all three `latest.pt` files.

### Four H100 GPUs on UCL SGE

Activate a compatible environment, enter this cloned repository, and use an
absolute cache path and a **new writable** run directory:

```bash
export FEATURE_CACHE=/absolute/path/to/teacher_cache
export FEATURE_RUN_ROOT=/absolute/path/to/compact_feature_runs
export FEATURE_ARM=geometry
qsub scripts/train_sge_4gpu.sh
```

The resource directives follow the existing UCL DMMD launcher. Check your account
and queue if they differ. This is one node with four GPUs, not four nodes. The
default update has four episodes globally: one GPU accumulates four episodes;
four GPUs process one episode each. The global episode order is independent of
rank count; floating-point results need not be identical.

### Logs, restart, and timing

```bash
tail -n 10 runs/geometry_seed42/train.jsonl
python train.py --cache data/teacher_cache --config configs/pilot.yaml \
  --arm geometry --workdir runs/geometry_seed42 --device cuda \
  --resume runs/geometry_seed42/latest.pt
python benchmark.py --cache data/teacher_cache \
  --checkpoint runs/geometry_seed42/latest.pt --device cuda \
  --output runs/geometry_seed42/benchmark.json
```

Restart requires the same cache, arm, executable source, and complete config/schedule. Timing covers
the frozen student and a generic input-gradient calculation; it is not a
measurement of the full Riesz loss or a teacher/student speedup ratio. Include
teacher caching, student pretraining, retained heads, and later generator cost
before claiming savings. Do not benchmark on a GPU shared with other jobs.

## Verification and limitations

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [validation results](validation/local_checks.json),
[limitations](docs/LIMITATIONS.txt), and the [original proposal](docs/DESIGN.txt).
The original proposal includes later gates that are not part of this release.
The real MAE-640 artifact and H100/4090 hardware were unavailable for local
verification; the synthetic checks do not establish representation usefulness.

## Small frozen-encoder audit

Before student training, [FROZEN_AUDIT.md](docs/FROZEN_AUDIT.md) describes a small
MAE/Inception comparison using independent real references and saved grids of
controlled latent perturbations. It includes an allocated one-H100 launcher.
This diagnostic measures response and input-backward cost, not generation quality.
