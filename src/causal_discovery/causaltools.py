import pandas as pd
import anndata as ad
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from graphviz import Digraph

gv_path = r"C:\Users\jenny\windows_10_cmake_Release_Graphviz-12.2.1-win32\Graphviz-12.2.1-win32\bin"
os.environ["PATH"] += os.pathsep + gv_path


class CausalDiscovery:
    def __init__(
        self,
        adata_path: str,
        gene_file: str,
        response_col: str = "group",
        output_folder: str = ".",
        bk=None,
    ):
        self.adata_path = adata_path
        self.gene_file = gene_file
        self.response_col = response_col
        self.out_dir = output_folder
        os.makedirs(self.out_dir, exist_ok=True)

        self.adata = ad.read_h5ad(self.adata_path)

        # subset by selected genes + build df
        self.load_genes()
        self.build_dataframe()

        # prepare background knowledge
        self.bk = bk
        self.build_background_knowledge()

    def load_genes(self):
        try:
            with open(self.gene_file) as f:
                self.genes = [g.strip() for g in f if g.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene file not found: {self.gene_file}")

    def build_dataframe(self):
        # full data matrix
        df = pd.DataFrame(
            (
                self.adata.X.toarray()
                if hasattr(self.adata.X, "toarray")
                else self.adata.X
            ),
            index=self.adata.obs_names,
            columns=self.adata.var_names,
        )
        # subset to only selected genes, drop all-zero cols
        df = df[self.genes]
        df = df.loc[:, (df != 0).any(axis=0)]

        resp = np.where(self.adata.obs[self.response_col] == "disease", 1, 0)
        df[self.response_col] = resp

        self.var_names = sorted(
            df.drop(columns=[self.response_col]).columns.tolist()
        )
        self.data_matrix = df[self.var_names].values

    def build_background_knowledge(self):
        group_node = GraphNode(self.response_col)
        self.nodes = [GraphNode(name) for name in self.var_names]
        for n in self.nodes:
            self.bk.add_forbidden_by_node(n, group_node)

    def run(
        self,
        method_func,
        method_kwargs: dict = None,
    ):

        kwargs = method_kwargs or {}
        # include the default arguments
        kwargs.update(
            {
                "data": self.data_matrix,
                "background_knowledge": self.bk,
            }
        )
        # call the causal method, unpack kwargs
        self.cg = method_func(**kwargs)
        self.Gnx = self.convert_to_nx()
        self.plot_graphviz(self.Gnx, edge_labels=None)
        return self.cg

    def convert_to_nx(self, adjacency_attr: str = "G"):

        if adjacency_attr == "G":
            A = getattr(self.cg, adjacency_attr).graph
        else:
            A = getattr(self.cg, adjacency_attr)
        n = A.shape[0]
        Gnx = nx.DiGraph()
        # add nodes
        for name in self.var_names:
            Gnx.add_node(name)
        # add edges
        unique_vals = set(np.unique(A))
        if unique_vals.issubset({-1, 0, 1}):
            for i in range(n):
                for j in range(n):
                    if A[i, j] == 1 and A[j, i] == -1:
                        Gnx.add_edge(self.var_names[i], self.var_names[j])
                    elif A[i, j] == 1 and A[j, i] == 1:
                        Gnx.add_edge(self.var_names[i], self.var_names[j])
                        Gnx.add_edge(self.var_names[j], self.var_names[i])
        else:
            for i in range(n):
                for j in range(n):
                    w = A[i, j]
                    if abs(w) >= 0.01:
                        Gnx.add_edge(
                            self.var_names[i], self.var_names[j], weight=w
                        )
        return Gnx

    def save_adjacency_matrix(self, filename: str, adjacency_attr: str = "G"):
        if adjacency_attr == "G":
            A = getattr(self.cg, adjacency_attr).graph
        else:
            A = getattr(self.cg, adjacency_attr)

        np.save(filename, A)
        print(f"Adjacency matrix saved.")

    def compute_hubs_bottlenecks(self, Gnx, top_n: int = 3):
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

    def plot_graphviz(
        self, Gnx: nx.DiGraph, edge_labels: dict = None, name=None
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
            fontsize="12",
            margin="0.3,0.3",
        )

        # Global edge styling
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

        # Add edges (with optional labels)
        for u, v in Gnx.edges():
            if edge_labels and (u, v) in edge_labels:
                label = str(edge_labels[(u, v)])
                dot.edge(u, v, label=label)
            else:
                dot.edge(u, v)

        # Output
        if not name:
            out_path = os.path.join(self.out_dir, "causal_graph")
        else:
            out_path = os.path.join(self.out_dir, "causal_graph" + name)
        dot.render(filename=out_path, cleanup=True)
        print(f"Saved Graphviz graph to {out_path}.png")
