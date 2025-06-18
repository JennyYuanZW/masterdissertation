# Load libraries
library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
library(ggplot2)

gene_symbols <- readLines("results/lasso/lasso_genes.txt")
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

?write.csv(as.data.frame(ego_bp), file = "GO_BP_enrichment_results_rf.csv")
png("GO_BP_barplot_top10.png", width = 900, height = 900, res = 150)
p <- barplot(
    ego_bp,
    showCategory = 10,
    drop = TRUE,
    order = TRUE, # turn on ordering
    decreasing = FALSE,
    title = "Top 10 Enriched GO BP Terms",
    font.size = 9,
    color = "p.adjust"
) +
    scale_fill_gradient(low = "red", high = "blue") # colour gradient by adjusted p-value
print(p)
dev.off()
