#!/usr/bin/env python3
"""Cell2location Visium spatial deconvolution pipeline (ish_dpn project).

Config-driven port of ishdpn/c2l_TEMPLATE.ipynb. Two-stage cell2location
workflow: (1) RegressionModel signature training on an snRNA-seq reference,
(2) Cell2location spatial deconvolution on concatenated Visium samples.
Both stages save/resume from disk so a re-run picks up where it left off.
"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

DEFAULTS = {
    "input_sn_counts": "input_sn/ref20k_counts.csv",
    "input_sn_meta": "input_sn/ref20k_meta.csv",
    "input_visium_dir": "input_visium/",
    "output_dir": ".",
    "sample_subset": None,  # null = use every sample dir under input_visium_dir
    "gene_filter": {"cell_count_cutoff": 5, "cell_percentage_cutoff2": 0.03, "nonz_mean_cutoff": 1.12},
    "ref_batch_key": "Box_ID",
    "ref_labels_key": "type",
    "ref_categorical_covariate_keys": ["Batch"],
    # cell type labels renamed so they don't collide with gene symbols downstream
    # (e.g. 'ATF3' the cell type vs. 'ATF3' the gene) -- see c2l_khadijahComp.ipynb cell 4
    "type_relabel": {"ATF3": "Inj.ATF3", "SMC": "SmMusc"},
    "max_epochs_ref": 250,
    "ref_export_num_samples": 1000,
    "ref_export_batch_size": 2500,
    "N_cells_per_location": 15,
    "detection_alpha": 20,
    "max_epochs_spatial": 5000,
    "spatial_batch_size": None,  # None = full-batch (production); set an int for bounded-memory runs
    "spatial_train_size": 1,
    "spatial_export_num_samples": 1000,
    "spatial_export_batch_size": 5000,
    "run_nmf_colocation": True,
    "nmf_n_fact_min": 3,
    "nmf_n_fact_max": 10,
    "nmf_n_restarts": 3,
    "compute_expected_per_cell_type": True,
    "samples_to_plot": None,  # null = plot every sample
    "force_ref": False,
    "force_spatial": False,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path):
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = dict(DEFAULTS)
    cfg.update(user_cfg)
    if "gene_filter" in user_cfg:
        merged = dict(DEFAULTS["gene_filter"])
        merged.update(user_cfg["gene_filter"])
        cfg["gene_filter"] = merged
    return cfg


def gpu_status():
    log(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, "
        f"cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        log(f"device: {torch.cuda.get_device_name(0)}, "
            f"free/total VRAM (MiB): {[x // 1024 // 1024 for x in torch.cuda.mem_get_info()]}")


# ---------------------------------------------------------------------------
# Stage 1: reference signature (RegressionModel)
# ---------------------------------------------------------------------------

def train_reference(cfg, out: Path):
    import cell2location
    from cell2location.models import RegressionModel
    from cell2location.utils.filtering import filter_genes

    ref_run_name = out / "reference_signatures"
    sc_h5ad = ref_run_name / "sc.h5ad"

    if sc_h5ad.exists() and not cfg["force_ref"]:
        log(f"reference signature already exists at {sc_h5ad}, loading (use force_ref: true to retrain)")
        adata_ref = sc.read_h5ad(sc_h5ad)
        return adata_ref

    log("loading snRNA reference counts/meta")
    counts = pd.read_csv(cfg["input_sn_counts"], index_col=0)
    meta = pd.read_csv(cfg["input_sn_meta"], index_col=0)
    adata_ref = sc.AnnData(counts.T)
    adata_ref.obs = meta

    labels_key = cfg["ref_labels_key"]
    type_relabel = cfg.get("type_relabel") or {}
    if type_relabel:
        adata_ref.obs[labels_key] = adata_ref.obs[labels_key].replace(type_relabel)
        log(f"relabeled {labels_key} values: {type_relabel}")

    # cell type labels become HDF5 group keys downstream (means_per_cluster_mu_fg_<label>);
    # h5py rejects "/" in keys, and several real DRG subtype labels use it as a marker-gene
    # separator (e.g. 'A-PEP.CHRNA7/SLC18A3') -- sanitize before it ever hits anndata.write()
    labels = adata_ref.obs[labels_key].astype(str)
    if labels.str.contains("/").any():
        offending = sorted(labels[labels.str.contains("/")].unique())
        adata_ref.obs[labels_key] = labels.str.replace("/", "_", regex=False)
        log(f"sanitized '/' -> '_' in {labels_key} labels (HDF5 key compatibility): {offending}")

    adata_ref.var["SYMBOL"] = adata_ref.var.index
    if hasattr(adata_ref, "raw") and adata_ref.raw is not None:
        del adata_ref.raw
    log(f"reference shape before gene filtering: {adata_ref.X.shape}")

    gf = cfg["gene_filter"]
    selected = filter_genes(
        adata_ref,
        cell_count_cutoff=gf["cell_count_cutoff"],
        cell_percentage_cutoff2=gf["cell_percentage_cutoff2"],
        nonz_mean_cutoff=gf["nonz_mean_cutoff"],
    )
    adata_ref = adata_ref[:, selected].copy()
    log(f"reference shape after gene filtering: {adata_ref.X.shape}")

    RegressionModel.setup_anndata(
        adata=adata_ref,
        batch_key=cfg["ref_batch_key"],
        labels_key=cfg["ref_labels_key"],
        categorical_covariate_keys=cfg["ref_categorical_covariate_keys"],
    )
    mod = RegressionModel(adata_ref)
    mod.view_anndata_setup()

    log(f"training RegressionModel: max_epochs={cfg['max_epochs_ref']}")
    t0 = time.time()
    mod.train(max_epochs=cfg["max_epochs_ref"])
    log(f"RegressionModel training done in {time.time() - t0:.1f}s")

    adata_ref = mod.export_posterior(
        adata_ref,
        sample_kwargs={"num_samples": cfg["ref_export_num_samples"], "batch_size": cfg["ref_export_batch_size"]},
    )
    ref_run_name.mkdir(parents=True, exist_ok=True)
    mod.save(str(ref_run_name), overwrite=True)
    adata_ref.write(sc_h5ad)
    log(f"saved reference signature model + adata to {ref_run_name}")
    return adata_ref


def build_signature_matrix(adata_ref):
    if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
        inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = adata_ref.var[
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    inf_aver.columns = adata_ref.uns["mod"]["factor_names"]
    return inf_aver


# ---------------------------------------------------------------------------
# Stage 2: spatial deconvolution (Cell2location)
# ---------------------------------------------------------------------------

def read_and_qc(sample_name, path):
    adata = sc.read_visium(str(Path(path) / sample_name), count_file="filtered_feature_bc_matrix.h5", load_images=True)
    adata.obs["sample"] = sample_name
    duplicated = adata.var_names.duplicated(keep=False)
    adata = adata[:, ~duplicated].copy()
    adata.var["SYMBOL"] = adata.var_names

    adata.X = adata.X.toarray()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.X = csr_matrix(adata.X)

    adata.var["MT_gene"] = [gene.startswith("MT-") for gene in adata.var["SYMBOL"]]
    adata.obs["MT_frac"] = adata[:, adata.var["MT_gene"].tolist()].X.sum(1).A.squeeze() / adata.obs["total_counts"]
    adata.obsm["MT"] = adata[:, adata.var["MT_gene"].values].X.toarray()
    adata = adata[:, ~adata.var["MT_gene"].values]

    adata.obs["sample"] = [str(i) for i in adata.obs["sample"]]
    adata.obs_names = adata.obs["sample"] + "_" + adata.obs_names
    adata.obs.index.name = "spot_id"
    return adata


def load_visium(cfg):
    visium_path = Path(cfg["input_visium_dir"])
    all_samples = sorted(d.name for d in visium_path.iterdir() if d.is_dir())
    sample_subset = cfg.get("sample_subset")
    sample_data = sample_subset if sample_subset else all_samples
    log(f"loading {len(sample_data)} Visium sample(s): {sample_data}")

    slides = [read_and_qc(s, cfg["input_visium_dir"]) for s in sample_data]
    # AnnData.concatenate() was removed in modern anndata -- anndata.concat() is the replacement;
    # label=/keys= are the direct equivalents of the old batch_key=/batch_categories=
    adata_vis = ad.concat(
        slides, join="inner", label="sample", keys=sample_data, index_unique=None, uns_merge="unique"
    )
    adata_vis.obsm["spatial"] = adata_vis.obsm["spatial"].astype(float)
    log(f"combined Visium adata shape: {adata_vis.shape}")
    return adata_vis


def fix_adata_vis_dtypes(adata_vis):
    int_cols = ["in_tissue", "array_row", "array_col", "n_genes_by_counts", "total_counts",
                "_indices", "_scvi_batch", "_scvi_labels"]
    for col in int_cols:
        if col in adata_vis.obs.columns:
            adata_vis.obs[col] = adata_vis.obs[col].astype(int)
    float_cols = ["log1p_n_genes_by_counts", "log1p_total_counts", "pct_counts_in_top_50_genes",
                  "pct_counts_in_top_100_genes", "pct_counts_in_top_200_genes", "pct_counts_in_top_500_genes",
                  "MT_frac"]
    for col in float_cols:
        if col in adata_vis.obs.columns:
            adata_vis.obs[col] = adata_vis.obs[col].astype(float)
    if "sample" in adata_vis.obs.columns:
        adata_vis.obs["sample"] = adata_vis.obs["sample"].astype(str)
    adata_vis.obsm["spatial"] = np.array(adata_vis.obsm["spatial"], dtype=np.float32)


def train_spatial(cfg, out: Path, adata_ref, inf_aver):
    import cell2location

    run_name = out / "cell2location_map"
    sp_h5ad = run_name / "sp.h5ad"

    if sp_h5ad.exists() and not cfg["force_spatial"]:
        log(f"spatial model already exists at {sp_h5ad}, loading (use force_spatial: true to retrain)")
        adata_vis = sc.read_h5ad(sp_h5ad)
        mod = cell2location.models.Cell2location.load(str(run_name), adata_vis)
        # mod.samples (needed by compute_expected_per_cell_type) is only populated by
        # export_posterior() -- .load() restores trained params but not the in-memory
        # posterior sample cache, so re-run it to repopulate mod.samples on resume.
        mod.export_posterior(
            adata_vis,
            sample_kwargs={"num_samples": cfg["spatial_export_num_samples"], "batch_size": cfg["spatial_export_batch_size"]},
        )
        return adata_vis, mod

    adata_vis = load_visium(cfg)

    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver_i = inf_aver.loc[intersect, :].copy()
    log(f"shared genes between reference signature and Visium data: {len(intersect)}")

    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

    torch.set_float32_matmul_precision("medium")
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver_i,
        N_cells_per_location=cfg["N_cells_per_location"],
        detection_alpha=cfg["detection_alpha"],
    )
    mod.view_anndata_setup()

    log(f"training Cell2location: max_epochs={cfg['max_epochs_spatial']}, "
        f"batch_size={cfg['spatial_batch_size']}, train_size={cfg['spatial_train_size']}")
    gpu_status()
    t0 = time.time()
    mod.train(
        max_epochs=cfg["max_epochs_spatial"],
        batch_size=cfg["spatial_batch_size"],
        train_size=cfg["spatial_train_size"],
    )
    log(f"Cell2location training done in {time.time() - t0:.1f}s")
    gpu_status()

    run_name.mkdir(parents=True, exist_ok=True)
    adata_vis = mod.export_posterior(
        adata_vis,
        sample_kwargs={"num_samples": cfg["spatial_export_num_samples"], "batch_size": cfg["spatial_export_batch_size"]},
    )
    mod.save(str(run_name), overwrite=True)

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    fix_adata_vis_dtypes(adata_vis)
    adata_vis.write(sp_h5ad)
    log(f"saved spatial model + adata to {run_name}")
    return adata_vis, mod


# ---------------------------------------------------------------------------
# Stage 3: exports (CSVs, clustering, colocation, plots)
# ---------------------------------------------------------------------------

def export_abundance_csvs(adata_vis, out: Path):
    keys = ["means_cell_abundance_w_sf", "stds_cell_abundance_w_sf", "q05_cell_abundance_w_sf", "q95_cell_abundance_w_sf"]
    for key in keys:
        df = pd.DataFrame(adata_vis.obsm[key], index=adata_vis.obs_names)
        dest = out / f"cell2location_{key}.csv"
        df.to_csv(dest)
        log(f"exported {dest}")


def export_rna_percentages(adata_ref, adata_vis, out: Path):
    cell_abundance = adata_vis.obsm["q05_cell_abundance_w_sf"]
    rna_content = adata_ref.varm["means_per_cluster_mu_fg"]

    cell_abundance_clean = cell_abundance.copy()
    cell_abundance_clean.columns = [c.replace("q05cell_abundance_w_sf_", "") for c in cell_abundance_clean.columns]
    rna_content_clean = rna_content.copy()
    rna_content_clean.columns = [c.replace("means_per_cluster_mu_fg_", "") for c in rna_content_clean.columns]

    common_cols = list(set(cell_abundance_clean.columns) & set(rna_content_clean.columns))
    if not common_cols:
        log("WARNING: no common cell types between abundance and RNA-content columns; skipping RNA percentage export")
        return

    cell_abundance_aligned = cell_abundance_clean[common_cols]
    rna_content_aligned = rna_content_clean[common_cols]
    rna_content_total = rna_content_aligned.sum(axis=0)
    rna_contribution = cell_abundance_aligned * rna_content_total
    rna_percentages = rna_contribution.div(rna_contribution.sum(axis=1), axis=0) * 100
    rna_percentages.index.name = "spot_id"

    rna_percentages.to_csv(out / "rna_percentages_by_cell_type.csv")
    pd.DataFrame({"cell_type": rna_content_total.index, "total_rna_content": rna_content_total.values}).to_csv(
        out / "rna_content_per_cell_type_total.csv", index=False
    )
    log(f"exported RNA percentage tables ({rna_percentages.shape[0]} spots x {rna_percentages.shape[1]} cell types)")


def export_region_clusters(adata_vis, out: Path, abundance_type="q05_cell_abundance_w_sf"):
    adata_vis.obs[adata_vis.uns["mod"]["factor_names"]] = adata_vis.obsm[abundance_type]
    sc.pp.neighbors(adata_vis, use_rep=abundance_type, n_neighbors=15)
    sc.tl.leiden(adata_vis, resolution=1.1)
    adata_vis.obs["region_cluster"] = adata_vis.obs["leiden"].astype("category")
    sc.tl.umap(adata_vis, min_dist=0.3, spread=1)

    cluster_export = pd.DataFrame({
        "barcode": adata_vis.obs.index,
        "region_cluster": adata_vis.obs["region_cluster"],
        "sample": adata_vis.obs["sample"],
    })
    dest = out / "region_cluster_labels.csv"
    cluster_export.to_csv(dest, index=False)
    log(f"exported {dest} ({cluster_export['region_cluster'].nunique()} region clusters, "
        f"{cluster_export['sample'].nunique()} samples)")


def run_nmf_colocation(cfg, adata_vis, run_name: Path):
    from cell2location import run_colocation

    res_dict, adata_vis = run_colocation(
        adata_vis,
        model_name="CoLocatedGroupsSklearnNMF",
        train_args={
            "n_fact": np.arange(cfg["nmf_n_fact_min"], cfg["nmf_n_fact_max"]),
            "sample_name_col": "sample",
            "n_restarts": cfg["nmf_n_restarts"],
        },
        model_kwargs={"alpha": 0.01, "init": "random", "nmf_kwd_args": {"tol": 0.000001}},
        export_args={"path": str(run_name / "CoLocatedComb") + "/"},
    )
    log(f"NMF colocation done, exported to {run_name / 'CoLocatedComb'}")
    return adata_vis


def compute_expected_expression(mod, adata_vis, run_name: Path, sp_h5ad: Path):
    expected_dict = mod.module.model.compute_expected_per_cell_type(mod.samples["post_sample_q05"], mod.adata_manager)
    for i, n in enumerate(mod.factor_names_):
        adata_vis.layers[n] = expected_dict["mu"][i]
    adata_vis.write(sp_h5ad)
    log(f"computed per-cell-type expected expression, re-saved {sp_h5ad}")
    return adata_vis


def plot_slide(adata_vis, sample_name, out: Path):
    import matplotlib as mpl

    mpl.use("Agg")
    from cell2location.utils import select_slide

    slide = select_slide(adata_vis, sample_name)
    cell_types_all = adata_vis.obsm["q05_cell_abundance_w_sf"].columns.str.replace("q05cell_abundance_w_sf_", "").tolist()

    with mpl.rc_context({"axes.facecolor": "black", "figure.figsize": [4.5, 5]}):
        fig = sc.pl.spatial(
            slide, cmap="magma", color=cell_types_all, ncols=4, size=1.3, img_key="hires",
            vmin=0, vmax="p99.2", return_fig=True, show=False,
        )
    dest = out / f"{sample_name}_deconv.pdf"
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"exported {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    log("=== dpnvisium: cell2location Visium deconvolution ===")
    gpu_status()

    adata_ref = train_reference(cfg, out)
    inf_aver = build_signature_matrix(adata_ref)

    adata_vis, mod = train_spatial(cfg, out, adata_ref, inf_aver)

    export_abundance_csvs(adata_vis, out)
    export_rna_percentages(adata_ref, adata_vis, out)
    export_region_clusters(adata_vis, out)

    run_name = out / "cell2location_map"
    if cfg["run_nmf_colocation"]:
        adata_vis = run_nmf_colocation(cfg, adata_vis, run_name)

    if cfg["compute_expected_per_cell_type"]:
        adata_vis = compute_expected_expression(mod, adata_vis, run_name, run_name / "sp.h5ad")

    samples_to_plot = cfg.get("samples_to_plot") or sorted(adata_vis.obs["sample"].unique().tolist())
    for s in samples_to_plot:
        plot_slide(adata_vis, s, out)

    log("=== done ===")


if __name__ == "__main__":
    sys.exit(main())
