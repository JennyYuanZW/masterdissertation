# Load libraries
library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
library(ggplot2)

gene_symbols <- readLines("results/rf/random_forest_genes.txt")
# Convert gene symbols to Entrez IDs
entrez_ids <- bitr(gene_symbols,
    fromType = "SYMBOL",
    toType = "ENTREZID",
    OrgDb = org.Hs.eg.db
)
# Check mapping results
print(entrez_ids)
# Perform GO enrichment analysis for Biological Processes (BP)
ego_bp <- enrichGO(
    gene = entrez_ids$ENTREZID,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2,
    readable = TRUE
)

write.csv(as.data.frame(ego_bp), file = "GO_BP_enrichment_results_rf.csv")
