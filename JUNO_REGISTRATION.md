# Juno Registration Instructions — dpnvisium pipeline

Step-by-step instructions to add `dpnvisium` and `dpnvisium-gpu` to the
`mwilde49/hpc` Hyperion Compute framework on Juno, following the exact
precedent set by `dconvatac`'s own `HPC_INTEGRATION_INSTRUCTIONS.md`.

**I (Claude Code) have read-only SSH access to Juno and no configured push
access for a new GitHub repo — everything below needs to be run by the human
user**, or handed to a future Claude Code session with write access. Every
file this doc tells you to copy already exists, finished and tested, in this
`dpnvisium/` repo — this doc is the deployment checklist, not a from-scratch
build.

---

## 0. Create and push the `mwilde49/dpnvisium` GitHub repo

```bash
cd /mnt/c/users/mwild/firebase2/dpnvisium   # or wherever you keep this repo
git add -A
git commit -m "Initial dpnvisium pipeline: script, container def, SLURM templates, registration docs"
gh repo create mwilde49/dpnvisium --private --source=. --remote=origin
git push -u origin master
git tag v1.0.0
git push origin v1.0.0
```

(Adjust remote name / branch name / visibility to your usual convention —
this mirrors how `dconvatac` was published.)

## 1. Register the submodule (on your `hpc` checkout, then push, then pull on Juno)

```bash
cd /mnt/c/users/mwild/firebase2/hpc      # your local hpc checkout
git submodule add https://github.com/mwilde49/dpnvisium containers/dpnvisium
cd containers/dpnvisium && git checkout v1.0.0 && cd ../..
git add .gitmodules containers/dpnvisium
git commit -m "Add dpnvisium submodule: Cell2Location Visium deconvolution pipeline"
git push
```

Then on Juno (via a session with write access — not available to this
inspect-only key):
```bash
cd /groups/tprice/pipelines
git pull
git submodule update --init containers/dpnvisium
```

## 2. Build the container (requires sudo or fakeroot)

```bash
cd containers/dpnvisium/container
sudo apptainer build ../dpnvisium_v1.0.0.sif apptainer.def
```

If building on a local machine (WSL2 Apptainer works as of 2026-07), transfer:
```bash
scp containers/dpnvisium/dpnvisium_v1.0.0.sif \
    juno:/groups/tprice/pipelines/containers/dpnvisium/
```

## 3. Copy SLURM templates and registration templates into the shared repo

These files already exist, finished, in this repo — copy them as-is (no
re-authoring needed, avoids drift between two copies of the same content):

```bash
cp slurm_templates/dpnvisium_gpu_slurm_template.sh \
   slurm_templates/dpnvisium_cpu_slurm_template.sh \
   $PROJECT_ROOT/slurm_templates/

mkdir -p $PROJECT_ROOT/templates/dpnvisium
cp templates/dpnvisium/config.yaml templates/dpnvisium/samplesheet.csv \
   $PROJECT_ROOT/templates/dpnvisium/

cp templates/dpnvisium_schema.yaml $PROJECT_ROOT/templates/schemas/dpnvisium.yaml
```

(`$PROJECT_ROOT` = `/groups/tprice/pipelines`, same as inside the SLURM
templates themselves.)

## 4. Modify `bin/lib/common.sh`

### 4a. In `PIPELINE_CONTAINERS`, add two entries after the `[dconvatac-gpu]` line:

```bash
    [dpnvisium]="containers/dpnvisium/dpnvisium_v1.0.0.sif"
    [dpnvisium-gpu]="containers/dpnvisium/dpnvisium_v1.0.0.sif"
```

### 4b. In `PIPELINE_TEMPLATES`, add two entries after the `[dconvatac-gpu]` line:

```bash
    [dpnvisium]="slurm_templates/dpnvisium_cpu_slurm_template.sh"
    [dpnvisium-gpu]="slurm_templates/dpnvisium_gpu_slurm_template.sh"
```

### 4c. In `KNOWN_PIPELINES`, append `dpnvisium dpnvisium-gpu`:

Current last line (verified in the live `bin/lib/common.sh`, line 64):
```bash
KNOWN_PIPELINES=(addone bulkrnaseq psoma virome cellranger cellranger-mkfastq cellranger-multi spaceranger xeniumranger sqanti3 wf-transcriptomes dconvatac dconvatac-gpu)
```
Change to:
```bash
KNOWN_PIPELINES=(addone bulkrnaseq psoma virome cellranger cellranger-mkfastq cellranger-multi spaceranger xeniumranger sqanti3 wf-transcriptomes dconvatac dconvatac-gpu dpnvisium dpnvisium-gpu)
```

## 5. Modify `bin/lib/validate.sh`

**Deviation from the dconvatac pattern:** dconvatac's `-gpu` validator adds an
extra check that `use_gpu: true` is set in the config, because dconvatac's own
pipeline script branches on a `use_gpu` config flag. dpnvisium has no such
flag — GPU vs. CPU is decided entirely by which SLURM template you submit
(`--nv`/`--gres=gpu` or not); the container and script are identical either
way, and torch auto-detects CUDA. So `_validate_dpnvisium_gpu` below is
intentionally identical to the base validator (no extra flag check) rather
than adding an unused `use_gpu` config field just for validator-pattern
parity.

### 5a. In the `validate_config` dispatcher `case` block, add two entries before the `*)` catch-all:

```bash
        dpnvisium)     _validate_dpnvisium "$config" errors ;;
        dpnvisium-gpu) _validate_dpnvisium "$config" errors ;;
```

### 5b. Add the `_validate_dpnvisium` function at the end of the file:

```bash
# ── dpnvisium validator ──────────────────────────────────────────────────────
_validate_dpnvisium() {
    local config="$1"
    local -n _errs=$2

    local required_keys=(input_sn_counts input_sn_meta input_visium_dir output_dir)
    for key in "${required_keys[@]}"; do
        if ! yaml_has "$config" "$key"; then
            _errs+=("Missing required key: $key")
        fi
    done

    local path_keys=(input_sn_counts input_sn_meta input_visium_dir)
    for key in "${path_keys[@]}"; do
        if yaml_has "$config" "$key"; then
            local val
            val=$(yaml_get "$config" "$key") || true
            if [[ -n "$val" && "$val" != __* && "$val" != /path/to/* && ! -e "$val" ]]; then
                _errs+=("Path does not exist for $key: $val")
            fi
        fi
    done

    for key in run_nmf_colocation compute_expected_per_cell_type force_ref force_spatial; do
        if yaml_has "$config" "$key"; then
            local val
            val=$(yaml_get "$config" "$key") || true
            case "$val" in
                true|false) ;;
                *) _errs+=("$key must be 'true' or 'false', got: $val") ;;
            esac
        fi
    done

    for key in N_cells_per_location detection_alpha max_epochs_ref max_epochs_spatial \
               ref_export_num_samples ref_export_batch_size spatial_train_size \
               spatial_export_num_samples spatial_export_batch_size; do
        if yaml_has "$config" "$key"; then
            local val
            val=$(yaml_get "$config" "$key") || true
            if [[ -n "$val" && ! "$val" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                _errs+=("$key must be a number, got: $val")
            fi
        fi
    done
    # spatial_batch_size deliberately not checked here -- null/blank is valid
    # (full-batch production training), not just "missing".
}
```

## 6. tjp-batch: deliberately NOT wired up (scope decision, not an oversight)

Unlike dconvatac (per-row: one SLURM job per spatial sample) or
bulkrnaseq/psoma/virome (per-sheet: one job, Nextflow parallelizes internally
across samplesheet rows), dpnvisium doesn't read a samplesheet at all —
`dpnvisium.py` auto-discovers every sample directory under
`input_visium_dir` and processes them together in a single job (that's
how the underlying Cell2location model works: it's trained jointly across all
concatenated samples, not per-sample). Neither the per-row nor per-sheet
`tjp-batch` pattern fits, so `_BATCH_PER_ROW` in `bin/tjp-batch` is left
unchanged and no `_gen_dpnvisium_config` generator is added. The `sample_subset`
config field is the mechanism if a subset run is ever needed — set it directly
in `config.yaml`, launched via `tjp-launch` as normal.
`bin/lib/samplesheet.sh`'s `_SAMPLESHEET_REQUIRED_COLS` registry is likewise
left untouched (confirmed by reading the file: only `tjp-batch`'s internals
reference it, so it's dead weight for a pipeline that doesn't do per-row/
per-sheet batching).

## 7. New file: `DPNVISIUM_HPC_GUIDE.md` (place at repo root, alongside `DCONVATAC_HPC_GUIDE.md`)

```markdown
# dpnvisium Pipeline — HPC Guide

Visium spatial transcriptomics deconvolution using
[Cell2Location](https://cell2location.readthedocs.io) for the ish_dpn
(Diabetic Peripheral Neuropathy) collaboration project. Container and
pipeline code: `mwilde49/dpnvisium` (submodule at `containers/dpnvisium/`).

Two-stage workflow: (1) RegressionModel signature training on an snRNA-seq
reference, (2) Cell2location spatial deconvolution jointly across all
concatenated Visium samples. See `ishdpn/c2l_TEMPLATE.ipynb` in the workspace
for the original notebook this pipeline was converted from.

---

## Setup

### 1. Pull the submodule
\`\`\`bash
cd /groups/tprice/pipelines
git submodule update --init containers/dpnvisium
\`\`\`

### 2. Build the container (requires sudo or fakeroot)
\`\`\`bash
cd containers/dpnvisium/container
sudo apptainer build ../dpnvisium_v1.0.0.sif apptainer.def
\`\`\`

### 3. Run \`tjp-setup\` (if not done)
\`\`\`bash
tjp-setup
# Creates /work/$USER/pipelines/dpnvisium/config.yaml
\`\`\`

---

## Single run (CPU, small/test configs only — dev partition, 2h limit)
\`\`\`bash
vi /work/$USER/pipelines/dpnvisium/config.yaml
tjp-launch dpnvisium
\`\`\`

## Production run (GPU — H100 partition, full-batch, all samples)
\`\`\`bash
tjp-launch dpnvisium-gpu
\`\`\`

The GPU template requests one NVIDIA H100 (80GB VRAM). This is deliberately
H100, not A30 (24GB) — production config trains with \`spatial_batch_size:
null\` (full-batch) across all 16 concatenated Visium samples (40,599
spots), which the dconvatac guide's own precedent already flags as exceeding
A30's 24GB for datasets this size. Set \`spatial_batch_size\` to a bounded
integer (e.g. 2048) in config.yaml if you need to run on A30 instead —
substitute the partition/gres lines in the GPU SLURM template.

---

## Config reference

| Key | Default | Description |
|-----|---------|-------------|
| \`input_sn_counts\` / \`input_sn_meta\` | **required** | snRNA-seq reference counts/metadata CSVs |
| \`input_visium_dir\` | **required** | Directory containing one subdirectory per Visium sample (10x Space Ranger output layout) |
| \`output_dir\` | **required** | Directory for outputs (created if absent) |
| \`sample_subset\` | \`null\` | Optional list to process only specific sample dirs |
| \`N_cells_per_location\` | \`15\` | Tissue-density hyperprior (DRG-specific — see source notebook) |
| \`detection_alpha\` | \`20\` | Human Visium value; use \`200\` for mouse |
| \`max_epochs_ref\` / \`max_epochs_spatial\` | \`250\` / \`5000\` | Training epochs, reference / spatial stage |
| \`spatial_batch_size\` | \`null\` | \`null\` = full-batch (production, needs H100); set an int to bound memory |
| \`run_nmf_colocation\` | \`true\` | NMF cellular-compartment colocation analysis |
| \`compute_expected_per_cell_type\` | \`true\` | Per-cell-type expected expression (needed for downstream NCEM) |
| \`force_ref\` / \`force_spatial\` | \`false\` | Retrain even if a saved model already exists (save/resume by default) |

Full field list: \`templates/dpnvisium/config.yaml\`.

---

## Outputs

Written to \`output_dir\`:
- \`reference_signatures/\` — RegressionModel + \`sc.h5ad\`
- \`cell2location_map/\` — Cell2location model, \`sp.h5ad\`, \`CoLocatedComb/\` (NMF)
- \`cell2location_{means,stds,q05,q95}_cell_abundance_w_sf.csv\`
- \`rna_percentages_by_cell_type.csv\`, \`rna_content_per_cell_type_total.csv\`
- \`region_cluster_labels.csv\`
- \`{sample}_deconv.pdf\` — one per Visium sample

---

## SLURM resources

| Pipeline | Partition | Time | CPUs | Memory | GPU |
|----------|-----------|------|------|--------|-----|
| \`dpnvisium\` | dev | 2h | 8 | 32 GB | — |
| \`dpnvisium-gpu\` | h100 | 12h | 16 | 128 GB | 1× H100 (80 GB) |

---

## References

- [Cell2location tutorial](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html)
- [Cell2location paper](https://doi.org/10.1038/s41587-021-01139-4)
- Source notebook: \`ishdpn/c2l_TEMPLATE.ipynb\`
```

## 8. Update `CLAUDE.md` and `README.md` (hpc repo)

Same pattern as dconvatac's own integration (see steps 12-13 of
`dconvatac/HPC_INTEGRATION_INSTRUCTIONS.md`): bump the pipeline count,
add a `dpnvisium` bullet to the pipeline list / README table, add a short
`## dpnvisium Pipeline` section referencing this doc and
`DPNVISIUM_HPC_GUIDE.md`.

## Verification checklist

```bash
tjp-test-suite --layer 1
tjp-test-suite --layer 2

source bin/lib/common.sh
is_known_pipeline dpnvisium
is_known_pipeline dpnvisium-gpu
get_slurm_template dpnvisium
get_slurm_template dpnvisium-gpu

tjp-validate dpnvisium
```

Consider also adding `bin/lib/tests/test_dpnvisium.sh` (Layer 1/2 assertions,
mirroring `bin/lib/tests/test_dconvatac.sh`) once the config schema above is
deployed — not included here to keep this handoff focused on what's needed
to get a real job launched first.
