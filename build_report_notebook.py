#!/usr/bin/env python3
"""Build dpnvisium_report.ipynb: a report notebook that mirrors the structure
of the original ishdpn/c2l_TEMPLATE.ipynb (same section order, same plots),
but LOADS a completed run's saved model+adata instead of training from
scratch -- so a finished run can be reviewed in the notebook form the lab is
used to, not just as raw CSVs.

Usage:
    python build_report_notebook.py --output-dir /path/to/completed/run/output_dir \
        [--samples-to-show S1-3 S2-3 S3-3] [--out dpnvisium_report.ipynb]

Then render to HTML the lab actually reads:
    jupyter nbconvert --to html --execute dpnvisium_report.ipynb
"""
import argparse
import json

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build(output_dir, samples_to_show, config_path):
    nb = nbf.v4.new_notebook()
    cells = []

    # --- Title / provenance -------------------------------------------------
    cells.append(md(
        "# dpnvisium Report — Cell2Location Visium Deconvolution\n\n"
        "ish_dpn (Diabetic Peripheral Neuropathy) project. This notebook mirrors "
        "`ishdpn/c2l_TEMPLATE.ipynb`'s section structure and plots, but **loads a "
        "completed run's saved model + adata rather than training from scratch** — "
        "this is a review/report artifact, not the training notebook. "
        "Source: [Cell2Location tutorial](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html) · "
        "pipeline: `dpnvisium.py`."
    ))

    # --- Parameters cell (first code cell, edit this to point at another run) --
    cells.append(md("#### RUN PARAMETERS ####\n\nEdit `OUTPUT_DIR` below to report on a different run."))
    cells.append(code(
        f"OUTPUT_DIR = {output_dir!r}\n"
        f"CONFIG_PATH = {config_path!r}\n"
        f"SAMPLES_TO_SHOW = {samples_to_show!r}  # subset for inline plots; every sample already has its own *_deconv.pdf in OUTPUT_DIR\n"
    ))
    cells.append(code(
        "import yaml\n"
        "from pathlib import Path\n\n"
        "with open(CONFIG_PATH) as f:\n"
        "    cfg = yaml.safe_load(f)\n\n"
        "print('=== Run configuration ===')\n"
        "for k, v in cfg.items():\n"
        "    print(f'{k:30s} {v}')\n"
    ))

    # --- Setup ----------------------------------------------------------------
    cells.append(md("#### SET UP PACKAGES ####"))
    cells.append(code(
        "import os\n"
        "import sys\n"
        "import scanpy as sc\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "import cell2location\n"
        "import matplotlib as mpl\n"
        "from matplotlib import rcParams\n"
        "import matplotlib.pyplot as plt\n"
        "rcParams['pdf.fonttype'] = 42\n\n"
        "print(torch.version.cuda)\n"
        "print(torch.cuda.is_available())\n\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "out = Path(OUTPUT_DIR)\n"
        "ref_run_name = out / 'reference_signatures'\n"
        "run_name = out / 'cell2location_map'\n"
    ))

    # --- Load reference signature ---------------------------------------------
    cells.append(md(
        "#### LOAD REFERENCE SIGNATURE ####\n\n"
        "Loads the already-trained `RegressionModel` (snRNA-seq reference signature) "
        "rather than retraining — see `dpnvisium.py`'s `train_reference()` for the "
        "training code this was produced by."
    ))
    cells.append(code(
        "adata_ref = sc.read_h5ad(ref_run_name / 'sc.h5ad')\n"
        "mod = cell2location.models.RegressionModel.load(str(ref_run_name), adata_ref)\n"
        "# .load() restores trained params but not the in-memory posterior sample\n"
        "# cache -- re-run export_posterior so QC/plotting methods that need\n"
        "# mod.samples work (same fix as dpnvisium.py's resume path).\n"
        "adata_ref = mod.export_posterior(adata_ref, sample_kwargs={'num_samples': 200, 'batch_size': 2500})\n"
        "print('Reference shape:', adata_ref.X.shape)\n"
        "mod.view_anndata_setup()\n"
    ))
    cells.append(code("mod.plot_QC()  # 2D histogram QC -- expect a 'noisy diagonal'"))

    cells.append(md("#### SIGNATURE MATRIX ####"))
    cells.append(code(
        "if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():\n"
        "    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'\n"
        "                                    for i in adata_ref.uns['mod']['factor_names']]].copy()\n"
        "else:\n"
        "    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'\n"
        "                                    for i in adata_ref.uns['mod']['factor_names']]].copy()\n"
        "inf_aver.columns = adata_ref.uns['mod']['factor_names']\n"
        "inf_aver.iloc[0:5, 0:5]\n"
    ))

    # --- Load spatial model -----------------------------------------------------
    cells.append(md(
        "#### LOAD SPATIAL MODEL ####\n\n"
        "Loads the already-trained `Cell2location` spatial deconvolution model — "
        "see `dpnvisium.py`'s `train_spatial()` for the training code."
    ))
    cells.append(code(
        "adata_vis = sc.read_h5ad(run_name / 'sp.h5ad')\n"
        "mod2 = cell2location.models.Cell2location.load(str(run_name), adata_vis)\n"
        "adata_vis = mod2.export_posterior(\n"
        "    adata_vis, sample_kwargs={'num_samples': 200, 'batch_size': 5000}\n"
        ")\n"
        "print('Spatial data shape:', adata_vis.shape)\n"
        "print('Samples:', sorted(adata_vis.obs[\"sample\"].unique().tolist()))\n"
        "mod2.view_anndata_setup()\n"
    ))
    cells.append(code(
        "mod2.plot_QC()\n"
        "fig = mod2.plot_spatial_QC_across_batches()\n"
    ))

    # --- Cell abundance results ---------------------------------------------
    cells.append(md("#### CELL ABUNDANCE RESULTS ####"))
    cells.append(code(
        "abundance_type = 'q05_cell_abundance_w_sf'\n"
        "q05 = pd.DataFrame(adata_vis.obsm[abundance_type], index=adata_vis.obs_names)\n"
        "q05.columns = [c.replace('q05cell_abundance_w_sf_', '') for c in q05.columns]\n"
        "print(f'{q05.shape[0]} spots x {q05.shape[1]} cell types')\n"
        "print('\\nTotal cell abundance per spot:')\n"
        "print(q05.sum(axis=1).describe())\n"
        "print('\\nTop 10 cell types by mean abundance:')\n"
        "q05.mean().sort_values(ascending=False).head(10)\n"
    ))

    cells.append(md("#### RNA PROPORTION ESTIMATION ####"))
    cells.append(code(
        "rna_pct_path = out / 'rna_percentages_by_cell_type.csv'\n"
        "if rna_pct_path.exists():\n"
        "    rna_pct = pd.read_csv(rna_pct_path, index_col=0)\n"
        "    print(rna_pct.describe().T[['mean', 'min', 'max']].sort_values('mean', ascending=False).head(10))\n"
        "else:\n"
        "    print('rna_percentages_by_cell_type.csv not found in', out)\n"
    ))

    # --- Region clustering ----------------------------------------------------
    cells.append(md("#### TISSUE REGIONS (Leiden clustering + UMAP) ####"))
    cells.append(code(
        "if 'X_umap' in adata_vis.obsm and 'region_cluster' in adata_vis.obs:\n"
        "    with mpl.rc_context({'axes.facecolor': 'white', 'figure.figsize': [8, 8]}):\n"
        "        sc.pl.umap(adata_vis, color=['region_cluster'], size=30,\n"
        "                   color_map='RdPu', ncols=2, legend_loc='on data', legend_fontsize=10)\n"
        "        sc.pl.umap(adata_vis, color=['sample'], size=30,\n"
        "                   color_map='RdPu', ncols=2, legend_fontsize=10)\n"
        "    print(adata_vis.obs['region_cluster'].value_counts().sort_index())\n"
        "else:\n"
        "    region_csv = out / 'region_cluster_labels.csv'\n"
        "    if region_csv.exists():\n"
        "        rc = pd.read_csv(region_csv)\n"
        "        print('UMAP not present in sp.h5ad -- showing region_cluster_labels.csv summary instead')\n"
        "        print(rc['region_cluster'].value_counts().sort_index())\n"
    ))

    # --- NMF colocation ---------------------------------------------------------
    cells.append(md(
        "#### CELL COMPARTMENTS (NMF colocation) ####\n\n"
        "Full results (heatmaps, per-factor spatial plots, stability plots) live in "
        "`cell2location_map/CoLocatedComb/` — this cell just summarizes what was produced."
    ))
    cells.append(code(
        "coloc_dir = run_name / 'CoLocatedComb'\n"
        "if coloc_dir.exists():\n"
        "    analyses = sorted(p.name for p in coloc_dir.iterdir() if p.is_dir())\n"
        "    print(f'{len(analyses)} NMF factor-count analyses found:')\n"
        "    for a in analyses:\n"
        "        print(' ', a)\n"
        "else:\n"
        "    print('No NMF colocation output found (run_nmf_colocation may have been false).')\n"
    ))

    # --- Per-slide exploration ---------------------------------------------------
    cells.append(md(
        "#### EXPLORE RESULTS PER SLIDE ####\n\n"
        "Showing `SAMPLES_TO_SHOW` inline; every sample in the run already has its own "
        "`{sample}_deconv.pdf` in `OUTPUT_DIR` regardless of what's shown here."
    ))
    cells.append(code(
        "from cell2location.utils import select_slide\n\n"
        "cell_types_all = adata_vis.obsm['q05_cell_abundance_w_sf'].columns.str.replace(\n"
        "    'q05cell_abundance_w_sf_', '').tolist()\n\n"
        "for sample_to_select in SAMPLES_TO_SHOW:\n"
        "    if sample_to_select not in adata_vis.obs['sample'].unique():\n"
        "        print(f'{sample_to_select} not in this run, skipping')\n"
        "        continue\n"
        "    slide = select_slide(adata_vis, sample_to_select)\n"
        "    print(f'--- {sample_to_select} ({slide.shape[0]} spots) ---')\n"
        "    with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': [4.5, 5]}):\n"
        "        sc.pl.spatial(slide, cmap='magma', color=cell_types_all, ncols=4, size=1.3,\n"
        "                      img_key='hires', vmin=0, vmax='p99.2')\n"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Completed run's output_dir")
    parser.add_argument("--config-path", required=True, help="config.yaml used for that run")
    parser.add_argument("--samples-to-show", nargs="*", default=["S1-3", "S2-3", "S3-3"])
    parser.add_argument("--out", default="dpnvisium_report.ipynb")
    args = parser.parse_args()

    notebook = build(args.output_dir, args.samples_to_show, args.config_path)
    with open(args.out, "w") as f:
        nbf.write(notebook, f)
    print(f"wrote {args.out}")
