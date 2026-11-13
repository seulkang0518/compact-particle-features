# Portable data and teacher cache

This first-stage pilot consumes **already generated latent states**. It does not
export samples from a generator, train that generator, or compute FID. Real MAE
weights, real latent arrays, and generated snapshots must be supplied separately.
Synthetic mode checks the pipeline and cannot establish representation quality.

## 1. Input files

Real examples use two NumPy files:

* `real_latents.npy`: floating point `[N,32,32,4]` in BHWC order.
* `real_labels.npy`: integer `[N]`, with ImageNet class indices `0..999`.
* Optional `real_ids.npy`: integer or Unicode `[N]` original-image IDs. All flips
  and other views of one original image must have the same ID. Without this
  file, row number is the image ID; use exactly one view per original image.

The latent must be a **sampled, scaled VAE state**, as supplied to the existing
generator/MAE: the original repository samples the VAE posterior and multiplies
by `0.18215`. This tool does not multiply again, decode RGB, or sample posterior
moments. Do not pass concatenated posterior means/log-variances. Do not pass
`NCHW` arrays.

Each generated snapshot is an NPZ with these keys:

```python
np.savez(
    "generated_early.npz",
    latents=generated_bhwc.astype("float32"),  # [N,32,32,4], already scaled
    labels=class_indices.astype("int64"),     # [N]
    noise_ids=np.asarray(noise_identifiers),  # [N], Unicode or integer
    checkpoint_id=np.asarray("state_00004000"),  # scalar Unicode
    cfg=np.asarray(1.19),                    # optional scalar or [N]
)
```

Use stable noise IDs across snapshots and CFG values. If you reuse the same
noise tensor at another class, retain the same noise ID. These IDs control the
held-out split. Object/pickled arrays are rejected. Duplicate
`(checkpoint_id, class, noise_id, cfg)` identities are rejected. Use at least two
different generator checkpoints; three or more are more informative. The exact
snapshot contents and source-file hashes are recorded for provenance.

## 2. Make a packet and check same-class support

```bash
python prepare_packet.py \
  --real-latents /path/to/real_latents.npy \
  --real-labels /path/to/real_labels.npy \
  --real-ids /path/to/real_ids.npy \
  --generated /path/to/generated_early.npz /path/to/generated_late.npz \
              /path/to/generated_heldout.npz \
  --heldout-checkpoint state_00014000 \
  --max-real-per-class 512 --max-generated-per-class 512 \
  --queries 8 --references 16 \
  --max-packet-gb 4 --max-input-gb 4 \
  --output data/imagenet_packet
```

`--heldout-checkpoint` must exactly equal one NPZ's `checkpoint_id`. Choose a
small subset of classes initially. Supply approximately **512 distinct real
images per class** and at least **256 distinct generated noise identities per
class per snapshot** when feasible. Actual split counts are hash determined;
these quantities are starting points rather than guaranteed support. The
per-class caps apply **after splitting**, separately to every split. Raising a
cap does not create missing examples.

The printed preflight report and `manifest.json` show per-split counts and
eligible classes for the requested query/reference count. All train, validation,
and audit splits must have at least one eligible class before teacher caching.
`Q=8,R=16` is the quick pilot. For a larger `Q=16,R=32` pilot, pass those values
to **both** commands and recheck eligibility. A total of 160 real images per
class generally leaves only about 24 validation images and cannot support
32 real references there. Sixteen generated examples per snapshot leave only
about two audit queries and cannot support the proposed audit.

Splits are deterministic `70/15/15` hashes of original real-image ID or global
noise ID. The audit contains only the held-out checkpoint **and** audit noise
IDs. Audit noise IDs are excluded from the other checkpoints; train/validation
noise IDs are excluded from the held-out checkpoint. This holds out both noise
identities and a generator snapshot. Real IDs are also disjoint across splits.
The same class can occur in every split. Episodes sample one class, then select
generated queries and real references without replacement within that episode.
Examples can recur across episodes of the same split.

The packet stores `inputs.npy` as float32 BCHW and records its checksum. No
input or existing output directory is overwritten. The default 4 GiB output
cap and free-space check guard against materializing a full ImageNet cache.
`--max-input-gb` also bounds each NPZ latent array's uncompressed memory use;
split large snapshots into several files if necessary.

## 3. Cache the frozen teacher

Real mode reuses a **trusted local** checkout of `riesz_flow_sophia` containing
`models/mae_model.py`, `models/hf.py`, and `utils/param_convert.py`. Install that
checkout's loader dependencies, including `einops`; loading a msgpack/JAX
artifact additionally requires its Flax dependencies. No teacher weights are
downloaded by this tool. The artifact directory must contain `metadata.json`
and `ema_params.pt` or `ema_params.msgpack` in the format already supported by
that checkout. A plain trainer `state_*.pt` is not a MAE artifact. An HF cache
root containing `models/mae/jax/mae_latent_640` can also be supplied.

```bash
python cache_teacher.py \
  --packet data/imagenet_packet \
  --source-repo /path/to/riesz_flow_sophia \
  --teacher-root /path/to/artifacts/models/mae/jax/mae_latent_640 \
  --teacher-name mae_latent_640 \
  --device cuda --microbatch 2 \
  --queries 8 --references 16 --positions 4 \
  --train-episodes 256 --val-episodes 64 --audit-episodes 64 \
  --max-cache-gb 4 --output data/mae640_pilot_cache
```

The loader rejects non-four-channel MAEs and `input_patch_size != 1` in v1.
Only the encoder is retained and moved onto the GPU. The original loader
briefly builds the full model on CPU, so sufficient host RAM is still required.
Teacher microbatching bounds activation memory; reduce `--microbatch` on a
24 GiB GPU if needed. This is a one-time offline cache cost. The program records
teacher source revision, dirty-file names, relevant code/weight hashes,
construction time, cache wall time, and actual cache bytes. The supplied real
MAE640 artifact still needs to be checked on the target machine.

Each episode contains `Q` generated states first, followed by `R` real states
of the **same class**. The five teacher stages are `conv1`, `layer1`, `layer2`,
`layer3`, and `layer4`; this pilot does not cache every residual block. At each
stage it stores sampled local vectors plus channelwise global mean and
population standard deviation (`sqrt(var + 1e-6)`). Tensors are shaped
`[positions, images, channels]`. Float16 targets reduce disk use; use
`--cache-dtype float32` if half targets overflow.

Local spatial indices use fixed disjoint half-pools: train/validation draw from
one half, audit from the other. `--positions` must fit in either half of every
stage (at most 8 for a deepest `4x4` map). This tests unseen **local feature
positions**; the global mean/std branches necessarily summarize all positions.
It does not hold out convolutional receptive fields or spatial statistics.

`index.json` is marked complete only after every episode is saved. It includes
stage channels/sizes, eligible classes, position pools, and episode SHA256s.
Payload files contain `inputs`, `positions`, `teacher_features`, `queries`,
`target_sizes`, and source metadata. Train/evaluate commands consume this cache.

## 4. CPU smoke cache without any external data

```bash
python cache_teacher.py --synthetic --output data/synthetic \
  --train-episodes 8 --val-episodes 2 --audit-episodes 2 \
  --queries 4 --references 4 --positions 2 --synthetic-width 4
```

The synthetic teacher is a fixed random convolutional network. Its metadata
explicitly marks the cache synthetic. Real and generated synthetic IDs and
noise IDs are disjoint across splits, and audit metadata uses a separate
synthetic checkpoint identity. Passing this smoke test verifies software
wiring only; it supplies no evidence about ImageNet generation or FID.
