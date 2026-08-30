# dpnvisium

Cell2Location Visium spatial transcriptomics deconvolution pipeline for the
**ish_dpn** project — the Diabetic Peripheral Neuropathy (DPN) collaboration
with Khadijah (see `firebase2/ishdpn/`). Converts the working, already-once-run
`ishdpn/c2l_TEMPLATE.ipynb` notebook into a reproducible, config-driven
pipeline, registered as a Hyperion Compute pipeline on Juno HPC
(`dpnvisium` / `dpnvisium-gpu`), following the same pattern as the
`dconvatac` pipeline (the other Cell2Location-on-Juno precedent in this
workspace).

Backs the `dpnabstract` Brain Summit 2026 poster's deconvolution results.

## What it does

Two-stage workflow, matching the source notebook exactly (same
`cell2location.models.RegressionModel` / `Cell2location` API calls, not a
third-party wrapper):

1. Train a `RegressionModel` cell-type signature on an snRNA-seq reference.
2. Train a `Cell2location` spatial model jointly across all concatenated
   Visium samples, export cell-type abundances, run Leiden region
   clustering + NMF colocation, and produce per-slide deconvolution plots.

Both stages save/resume from disk — re-running with the same `output_dir`
picks up where it left off unless `force_ref`/`force_spatial` is set.

## Running it

### Standalone (local, or any machine with the deps installed)

```bash
python dpnvisium.py --config my_config.yaml
```

See `templates/dpnvisium/config.yaml` for every field and its default
(mirrors the source notebook's parameters — `N_cells_per_location: 15`,
`detection_alpha: 20`, `max_epochs_spatial: 5000`, `spatial_batch_size: null`
for full-batch production training).

`smoke_test.py` is a separate, synthetic-data-only script that exercises the
same cell2location API surface with tiny fake data and 5 epochs — run it
first in any new environment to confirm the torch/cuda/cell2location install
actually works before pointing the real pipeline at real data.

### On Juno (production)

Not deployed yet — this repo is the finished, locally-validated pipeline
code, ready to hand off. See **`JUNO_REGISTRATION.md`** for the full
step-by-step deployment checklist (submodule registration, container build,
`bin/lib/common.sh`/`validate.sh` diffs, everything needed to make
`tjp-launch dpnvisium-gpu` work). That doc also explains why: full
production training is deliberately full-batch across all ~40,600 spots
(all 16 Visium samples concatenated), which needs H100-class VRAM — not
something reproducible on a laptop GPU or even Juno's A30 partition.

## Repo layout

- `dpnvisium.py` — the pipeline script
- `smoke_test.py` — environment sanity check (synthetic data)
- `container/apptainer.def` — Apptainer container definition
- `slurm_templates/` — GPU (H100) and CPU (dev-partition) SLURM templates
- `templates/dpnvisium/` — config.yaml + samplesheet.csv templates for
  Hyperion Compute registration
- `templates/dpnvisium_schema.yaml` — config validator schema
- `JUNO_REGISTRATION.md` — deployment checklist (needs a write-access session)
- `DPNVISIUM_HPC_GUIDE.md` content — embedded in `JUNO_REGISTRATION.md` §7,
  to be placed at the `hpc` repo root alongside `DCONVATAC_HPC_GUIDE.md`
  during deployment

## Related

- `firebase2/ishdpn/` — the source notebook, input data, and prior results
- `firebase2/dpnabstract/` — the Brain Summit 2026 poster this backs
- `firebase2/dconvatac/` — the sibling Cell2Location-on-Juno pipeline
  (spatial ATAC, not Visium) this was modeled on
- `firebase2/jayden/`, `firebase2/khadijah/mlc/` — related DRG spatial
  deconvolution work in this workspace
