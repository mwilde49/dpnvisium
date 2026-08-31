#!/usr/bin/env Rscript
# One-time extraction: pull the per-spot NGN/neuron classification (DPN.type)
# out of Khadijah's legacy `vis_barcodes 1.rds` Seurat object (from MLC.zip)
# into a lightweight CSV, so the reproduction Rmd doesn't need to load a
# 112MB Seurat object (and that file never gets committed to git).
#
# Usage: Rscript extract_legacy_classification.R <path/to/vis_barcodes 1.rds> <output_csv>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript extract_legacy_classification.R <vis_barcodes.rds> <output_csv>")
}
input_rds <- args[1]
output_csv <- args[2]

suppressMessages(library(Seurat))

x <- readRDS(input_rds)
md <- x@meta.data

out <- data.frame(
  sample = md$sample,
  barcode = md$origBarcode,
  DPN = md$DPN,
  DPN.type = md$DPN.type,
  type = md$type,
  stringsAsFactors = FALSE
)

write.csv(out, output_csv, row.names = FALSE)
cat(sprintf("Wrote %d rows to %s\n", nrow(out), output_csv))
cat("\nDPN.type distribution:\n")
print(table(out$DPN.type, useNA = "always"))
