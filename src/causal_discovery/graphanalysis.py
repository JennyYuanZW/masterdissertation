import numpy as np
import pandas as pd
import networkx as nx
import itertools
import anndata as ad
from graphviz import Digraph


def load_adjacencies(file_paths):
    adjs = []
    for path in file_paths:
        mat = np.load(path)
        adjs.append(mat)
    return adjs


def consensus_graph(adjs):
    bin_adjs = [(adj != 0).astype(int) for adj in adjs]
    stacked = np.stack(bin_adjs, axis=0)
    intersection = np.all(stacked == 1, axis=0).astype(int)
    print(intersection)
    return intersection


def structural_distance(A, B):
    A_binary = (np.array(A) != 0).astype(int)
    B_binary = (np.array(B) != 0).astype(int)
    print(np.sum(A_binary))
    print(np.sum(B_binary))

    diff = A_binary - B_binary
    total_differences = np.sum(np.abs(diff))

    reversals_matrix = A_binary * B_binary.T
    num_reversals = np.sum(reversals_matrix)

    shd = total_differences - num_reversals

    return int(total_differences)


def mat_to_nx(A, node_labels):
    n = A.shape[0]
    G = nx.DiGraph()
    for name in node_labels:
        G.add_node(name)
    unique_vals = set(np.unique(A))
    for i in range(n):
        for j in range(n):
            w = A[i, j]
            if w == 1:
                G.add_edge(node_labels[i], node_labels[j])

    return G


def plot_graphviz(
    Gnx: nx.DiGraph, out_path, edge_labels: dict = None, name=None
):
    dot = Digraph(comment="Causal Graph", format="png")
    dot.graph_attr["dpi"] = "200"
    dot.attr(rankdir="TB", size="64,80")  # top→bottom layout

    # Global node styling
    dot.attr(
        "node",
        shape="ellipse",
        style="filled",
        fillcolor="white",
        fontname="Helvetica",
        fontsize="8",
        margin="0.3,0.3",
    )

    # Global edge styling
    dot.attr(
        "edge",
        color="black",
        arrowsize="1.2",
        fontname="Helvetica",
        fontsize="8",
    )

    # Add nodes
    for node in Gnx.nodes():
        dot.node(node)

    # Add edges (with optional labels)
    for u, v in Gnx.edges():
        if edge_labels and (u, v) in edge_labels:
            label = str(edge_labels[(u, v)])
            dot.edge(u, v, label=label)
        else:
            dot.edge(u, v)

    dot.render(filename=out_path, cleanup=True)
    print(f"Saved Graphviz graph to {out_path}.png")


def top_edges_single(A, labels, N=10):
    """Return top-N edges by absolute weight from a single adjacency matrix A."""
    edges = [
        ((labels[i], labels[j]), abs(A[i, j]))
        for i, j in itertools.product(range(A.shape[0]), range(A.shape[1]))
        if A[i, j] != 0
    ]
    edges.sort(key=lambda x: x[1], reverse=True)
    return edges[:N]


def top_edges_combined(A1, A2, labels, N=10):
    """Return top-N edges by sum of absolute weights from two adjacency matrices."""
    edges = []
    for i, j in itertools.product(range(A1.shape[0]), range(A1.shape[1])):
        w1 = A1[i, j]
        w2 = A2[i, j]
        total = abs(w1) + abs(w2)
        if total != 0:
            edges.append(((labels[i], labels[j]), total, w1, w2))
    edges.sort(key=lambda x: x[1], reverse=True)
    return edges[:N]


if __name__ == "__main__":
    paths = [
        "results/rf/rf_GNN.npy",
        "results/lasso/lasso_IT.npy",
        "results/rf/rf_LINGAM.npy",
    ]
    GNN = np.load(paths[0])
    IT = np.load(paths[1])
    IT = (np.array(IT) == 1).astype(int)
    LINGAM = np.load(paths[1])
    adjs = load_adjacencies(paths)
    intersection = consensus_graph(adjs)

    shd_mat1 = structural_distance(IT, GNN)
    shd_mat2 = structural_distance(LINGAM, GNN)
    shd_mat3 = structural_distance(LINGAM, IT)
    adata = ad.read_h5ad("data/adata_balanced.h5ad")

    df = pd.DataFrame(
        (adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X),
        index=adata.obs_names,
        columns=adata.var_names,
    )

    with open("results/rf/random_forest_genes.txt") as f:
        genes = [g.strip() for g in f if g.strip()]

    df = df[genes]
    df = df.loc[:, (df != 0).any(axis=0)]
    resp = np.where(adata.obs["group"] == "disease", 1, 0)
    df["group"] = resp

    node_labels = sorted(df.drop(columns=["group"]).columns.tolist())
    G = mat_to_nx(intersection, node_labels)
    connected_nodes = [n for n, d in G.degree() if d > 0]
    G_conn = G.subgraph(connected_nodes).copy()
    num_edges = G_conn.number_of_edges()
    print(f"Number of edges in connected graph: {num_edges}")

    plot_graphviz(G_conn, "rf_consensus")
    print("Pairwise SHD1:\n", shd_mat1)
    print("Pairwise SHD2:\n", shd_mat2)
    print("Pairwise SHD3:\n", shd_mat3)
    # --- Print top-10 per graph ---
    print("Top 10 edges in GNN (by |weight|):")
    for (u, v), w in top_edges_single(GNN, node_labels, 10):
        print(f"  {u} → {v}: |{w:.4f}|")

    print("\nTop 10 edges in LiNGAM (by |weight|):")
    for (u, v), w in top_edges_single(LINGAM, node_labels, 10):
        print(f"  {u} → {v}: |{w:.4f}|")

    # --- Print top-10 by combined weight ---
    print("\nTop 10 edges by |GNN| + |LiNGAM|:")
    for (u, v), total, w1, w2 in top_edges_combined(
        GNN, LINGAM, node_labels, 10
    ):
        print(
            f"  {u} → {v}: |GNN|={abs(w1):.4f}, |LiNGAM|={abs(w2):.4f}, sum={total:.4f}"
        )
