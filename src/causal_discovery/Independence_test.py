import pandas as pd
import anndata as ad
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.linear_model import LinearRegression
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode


class CausalDiscovery:
    def __init__(
        self,
        adata_path: str,
        gene_file: str,
        response_col: str = "group",
        output_folder: str = ".",
    ):
        """
        adata_path:      path to your .h5ad
        gene_file:       one-gene-per-line text file
        response_col:    obs column in adata giving group labels
        output_folder:   where to dump plots/logs
        """
        self.adata_path = adata_path
        self.gene_file = gene_file
        self.response_col = response_col
        self.out_dir = output_folder
        os.makedirs(self.out_dir, exist_ok=True)

        # load AnnData
        self.adata = ad.read_h5ad(self.adata_path)

        # subset by selected genes + build df
        self._load_genes()
        self._build_dataframe()

        # prepare background knowledge
        self._build_background_knowledge()

    def _load_genes(self):
        try:
            with open(self.gene_file) as f:
                self.genes = [g.strip() for g in f if g.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene file not found: {self.gene_file}")

    def _build_dataframe(self):
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
        # response
        resp = np.where(self.adata.obs[self.response_col] == "disease", 1, 0)
        df[self.response_col] = resp
        # store
        self.data_matrix = df.drop(columns=[self.response_col]).values
        self.var_names = df.drop(columns=[self.response_col]).columns.tolist()

    def _build_background_knowledge(self):
        bk = BackgroundKnowledge()
        group_node = GraphNode(self.response_col)
        self.nodes = [GraphNode(name) for name in self.var_names]
        for n in self.nodes:
            bk.add_forbidden_by_node(n, group_node)
        self.bk = bk

    def run(
        self,
        method_func,
        method_kwargs: dict = None,
    ):
        """
        method_func:    a function like `pc(data_matrix, **kwargs, background_knowledge=self.bk)`
        method_kwargs:  dict of extra kwargs for that function
        """
        kwargs = method_kwargs or {}
        kwargs.update(
            {
                "data": self.data_matrix,
                "background_knowledge": self.bk,
            }
        )
        # call the causal method
        self.cg = method_func(**kwargs)
        return self.cg

    def convert_to_nx(self, adjacency_attr: str = "G"):
        """
        Takes the causal-graph object returned by run() and converts to nx.DiGraph.
        Defaults to using cg.G as the adjacency matrix.
        """
        A = getattr(self.cg, adjacency_attr).graph
        n = A.shape[0]
        Gnx = nx.DiGraph()
        # add nodes
        for name in self.var_names:
            Gnx.add_node(name)
        # add edges
        for i in range(n):
            for j in range(n):
                if A[i, j] == 1 and A[j, i] == -1:
                    Gnx.add_edge(self.var_names[i], self.var_names[j])
                elif A[i, j] == 1 and A[j, i] == 1:
                    Gnx.add_edge(self.var_names[i], self.var_names[j])
                    Gnx.add_edge(self.var_names[j], self.var_names[i])
        return Gnx

    def plot_graph(self, Gnx, figsize=(20, 18), **draw_kwargs):
        """
        Draws the NetworkX DiGraph.
        """
        pos = nx.spring_layout(Gnx)
        plt.figure(figsize=figsize)
        nx.draw(
            Gnx,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            arrows=True,
            arrowstyle="->",
            arrowsize=15,
            **draw_kwargs,
        )
        plt.title("Causal Graph")
        plt.axis("off")
        plt.tight_layout()
        fig_path = os.path.join(self.out_dir, "causal_graph.png")
        plt.savefig(fig_path)
        plt.show()
        print(f"Saved causal graph to {fig_path}")


adata_disease = ad.read_h5ad("adata_balanced.h5ad")


def read_gene_file(filepath):
    try:
        with open(filepath, "r") as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        print(f"Warning: {filepath} not found.")
        return set()


file1_path = "selected_genes_rf.txt"
rf_genes = read_gene_file(file1_path)
rf_genes = list(rf_genes)
df = pd.DataFrame(
    adata_disease.X,
    index=adata_disease.obs_names,
    columns=adata_disease.var_names,
)
select = df[rf_genes]

select = select.loc[:, (select != 0).any(axis=0)]
response = np.where(adata_disease.obs["group"] == "disease", 1, 0)
select["group"] = response

# Ensure the 'group' column exists
group_col = "group"
if group_col not in select.columns:
    raise ValueError(
        f"Column '{group_col}' not found. Available columns: {df.columns}"
    )

# Extract causal variables and their names
data_matrix = select.drop(columns=[group_col]).values
var_names = select.drop(columns=[group_col]).columns.tolist()

# Create nodes for causal discovery
nodes = [GraphNode(name) for name in var_names]

# Initialize background knowledge
bk = BackgroundKnowledge()

# Define 'group' as exogenous (no incoming edges)
group_node = GraphNode(group_col)
for node in nodes:
    bk.add_forbidden_by_node(
        node, group_node
    )  # Prevent edges pointing to 'group'

# Run PC algorithm with background knowledge
cg = pc(
    data_matrix,
    alpha=0.1,
    indep_test="fisherz",
    background_knowledge=bk,
    verbose=False,
)


def causal_graph_to_nx(causal_graph, labels):
    """
    Convert a causal graph (with an adjacency matrix) to a NetworkX DiGraph.
    Assumes causal_graph.graph is a square matrix where:
      - For a directed edge i -> j:
          causal_graph.graph[i, j] == 1 and causal_graph.graph[j, i] == -1
      - For a bidirected edge (i <-> j):
          causal_graph.graph[i, j] == causal_graph.graph[j, i] == 1
    """
    A = causal_graph.graph  # Adjacency matrix
    n = A.shape[0]
    G_nx = nx.DiGraph()

    # Add nodes with labels.
    for i in range(n):
        G_nx.add_node(labels[i])

    # Add directed edges.
    for i in range(n):
        for j in range(n):
            # Check for i -> j.
            if A[i, j] == 1 and A[j, i] == -1:
                G_nx.add_edge(labels[i], labels[j])
            # Optionally, handle bidirected edges (if present)
            elif A[i, j] == 1 and A[j, i] == 1:
                G_nx.add_edge(labels[i], labels[j])
                G_nx.add_edge(labels[j], labels[i])
    return G_nx


# Convert the causal graph to a NetworkX graph.
nx_graph = causal_graph_to_nx(cg.G, var_names)

# Verify nodes and edges
print("Nodes:", list(nx_graph.nodes()))
print("Edges:", list(nx_graph.edges()))

# Compute a layout and draw the graph.
pos = nx.spring_layout(nx_graph)
plt.figure(figsize=(20, 18))
nx.draw(
    nx_graph,
    pos,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    arrows=True,
    arrowstyle="->",
    arrowsize=15,
)
plt.title("Causal Graph")
plt.axis("off")
plt.show()
