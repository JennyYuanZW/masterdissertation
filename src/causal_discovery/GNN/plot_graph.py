import os
import numpy as np
import pandas as pd
import networkx as nx
from graphviz import Digraph


class CausalGraphPlotter:
    def __init__(self, path, threshold=0.01, var_names=None, out_dir="."):
        self.var_names = var_names
        self.out_dir = out_dir
        self.threshold = threshold
        self.adjacency_input = self.file_to_adjacency(path)

    def load_weight_matrix(self, path):
        return np.loadtxt(path)

    def to_adjacency_matrix(self, weight_mat):
        thresh_mat = np.where(
            np.abs(weight_mat) > self.threshold, weight_mat, 0
        )
        return thresh_mat

    def save_adjacency_npz(self, path, npy_path):
        adjacency_matrix = self.file_to_adjacency(path)
        np.save(npy_path, adjacency_matrix)
        print("Adjacency matrix saved")

    def file_to_adjacency(self, path):
        W = self.load_weight_matrix(path)
        return self.to_adjacency_matrix(W)

    def convert_to_nx(self):
        A = self.adjacency_input
        names = self.var_names
        n = A.shape[0]
        Gnx = nx.DiGraph()
        Gnx.add_nodes_from(names)

        unique_vals = set(np.unique(A))
        for i in range(n):
            for j in range(n):
                w = A[i, j]
                if abs(w) >= 0.01:
                    Gnx.add_edge(names[i], names[j], weight=w)

        self.edge_labels = {
            (names[i], names[j]): f"{A[i, j]:.2f}"
            for i in range(n)
            for j in range(n)
            if A[i, j] != 0
        }
        return Gnx

    def compute_hubs_bottlenecks(self, Gnx, top_n: int = 10):
        # Degree‐based hubs
        out_deg_centrality = nx.out_degree_centrality(Gnx)
        top_hubs = [
            name
            for name, _ in sorted(
                out_deg_centrality.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
        ]
        # Bottlenecks
        btw_centrality = nx.betweenness_centrality(Gnx, normalized=True)
        top_bottlenecks = [
            name
            for name, _ in sorted(
                btw_centrality.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
        ]

        return {"hubs": top_hubs, "bottlenecks": top_bottlenecks}

    def plot_graphviz(self, Gnx: nx.DiGraph, name=None):
        dot = Digraph(comment="Causal Graph", format="png")
        dot.graph_attr["dpi"] = "200"
        dot.attr(rankdir="TB", size="64,80")

        # Node styling
        dot.attr(
            "node",
            shape="ellipse",
            style="filled",
            fillcolor="white",
            fontname="Helvetica",
            fontsize="12",
            margin="0.3,0.3",
        )

        # Edge styling
        dot.attr(
            "edge",
            color="black",
            arrowsize="1.2",
            fontname="Helvetica",
            fontsize="12",
        )

        # Add nodes
        for node in Gnx.nodes():
            dot.node(node)

        # Add edges
        for u, v, data in Gnx.edges(data=True):
            label = None
            if self.edge_labels and (u, v) in self.edge_labels:
                label = str(self.edge_labels[(u, v)])
            elif "weight" in data:
                label = f"{data['weight']:.2f}"
            dot.edge(u, v, label=label if label else "")

        # Render output
        fname = "causal_graph" + (f"_{name}" if name else "")
        out_path = os.path.join(self.out_dir, fname)
        dot.render(filename=out_path, cleanup=True)
        print(f"Saved Graphviz graph to {out_path}.png")


def main():
    var_names = pd.read_csv("data/gene_expression_rf.csv").columns.tolist()[1:]
    causalgraph = CausalGraphPlotter("predG_rf.txt", var_names=var_names)
    causalgraph.save_adjacency_npz("predG_rf.txt", "GNN")
    Gnx = causalgraph.convert_to_nx()
    print(causalgraph.compute_hubs_bottlenecks(Gnx))
    causalgraph.plot_graphviz(Gnx)


if __name__ == "__main__":
    main()
