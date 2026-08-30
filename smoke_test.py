#!/usr/bin/env python3
"""Tier 0 smoke test: synthetic data through the full cell2location API surface.

Not connected to real ish_dpn data at all -- purpose is purely to prove the
installed torch/cuda/cell2location/scvi-tools combination works end to end
(imports, setup_anndata, .train(), .export_posterior()) before touching real
data or the real dpnvisium.py pipeline.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_synthetic_ref(n_cells=200, n_genes=50, n_types=5, seed=42):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = sc.AnnData(counts)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.var["SYMBOL"] = adata.var_names
    adata.obs["HighName"] = rng.choice([f"type{i}" for i in range(n_types)], size=n_cells)
    adata.obs["sample"] = rng.choice(["s1", "s2"], size=n_cells)
    adata.obs["Batch"] = rng.choice(["b1", "b2"], size=n_cells)
    return adata


def make_synthetic_visium(n_spots=100, n_genes=50, seed=42):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=3.0, size=(n_spots, n_genes)).astype(np.float32)
    adata = sc.AnnData(counts)
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obs["sample"] = "s1"
    adata.obsm["spatial"] = rng.uniform(0, 1000, size=(n_spots, 2))
    return adata


def main():
    log(f"python {sys.version}")
    import cell2location
    log(f"cell2location {cell2location.__version__}")
    log(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        log(f"device: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        log(f"GPU tensor matmul OK, result sum: {y.sum().item():.2f}")

    from cell2location.models import RegressionModel

    log("--- stage 1: synthetic RegressionModel ---")
    adata_ref = make_synthetic_ref()
    RegressionModel.setup_anndata(
        adata=adata_ref, batch_key="sample", labels_key="HighName", categorical_covariate_keys=["Batch"]
    )
    mod = RegressionModel(adata_ref)
    mod.train(max_epochs=5)
    adata_ref = mod.export_posterior(adata_ref, sample_kwargs={"num_samples": 20, "batch_size": 50})
    log("RegressionModel smoke test OK")

    inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
    ].copy()
    inf_aver.columns = adata_ref.uns["mod"]["factor_names"]

    log("--- stage 2: synthetic Cell2location ---")
    adata_vis = make_synthetic_visium()
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")
    mod2 = cell2location.models.Cell2location(
        adata_vis, cell_state_df=inf_aver, N_cells_per_location=15, detection_alpha=20
    )
    mod2.train(max_epochs=5, batch_size=None, train_size=1)
    adata_vis = mod2.export_posterior(adata_vis, sample_kwargs={"num_samples": 20, "batch_size": 50})
    log("Cell2location smoke test OK")

    log("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
