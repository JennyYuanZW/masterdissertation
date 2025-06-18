from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from causaltools import CausalDiscovery
import networkx as nx
import matplotlib.pyplot as plt


def main(gene_file, output_folder):
    causal_data = CausalDiscovery(
        adata_path="data/adata_balanced.h5ad",
        gene_file=gene_file,
        response_col="group",
        output_folder=output_folder,
        bk=BackgroundKnowledge(),
    )

    cg = causal_data.run(
        pc, {"alpha": 0.05, "indep_test": "fisherz", "stable": True}
    )
    causal_data.save_adjacency_matrix("IT", "G")
    print(cg.G.graph)
    print(causal_data.compute_hubs_bottlenecks(causal_data.Gnx, 5))


if __name__ == "__main__":
    main("results/gb/gb_genes.txt", "results/gb")
