from causallearn.search.FCMBased import lingam
from causaltools import CausalDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def main(gene_file, output_folder):
    causal_data = CausalDiscovery(
        adata_path="data/adata_balanced.h5ad",
        gene_file=gene_file,
        response_col="group",
        output_folder=output_folder,
        bk=BackgroundKnowledge(),
    )
    data = causal_data.data_matrix
    model = lingam.ICALiNGAM(None, 10000)
    causal_data.cg = model.fit(data)
    Gnx = causal_data.convert_to_nx("adjacency_matrix_")
    column_names = causal_data.var_names
    num_vars = len(column_names)
    A = causal_data.cg.adjacency_matrix_
    edge_labels = {
        (column_names[i], column_names[j]): f"{A[i, j]:.2f}"
        for i in range(num_vars)
        for j in range(num_vars)
        if A[i, j] != 0
    }
    causal_data.plot_graphviz(Gnx, edge_labels=edge_labels, name="LINGAM2")
    print(causal_data.compute_hubs_bottlenecks(Gnx, 3))


if __name__ == "__main__":
    main("results/gb/gb_genes.txt", "results/gb")
