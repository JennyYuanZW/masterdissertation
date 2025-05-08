import os
import logging
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


class RobustFeatureSelector:
    def __init__(
        self,
        model,
        model_name: str,
        data_path: str = "data/adata_balanced.h5ad",
        output_folder: str = "results",
        log_file: str = None,
        sample_size: int = 1000,
        num_iterations: int = 20,
        importance_threshold: float = 1e-5,
    ):
        self.model = model
        self.name = model_name
        self.data_path = data_path
        self.out_dir = output_folder
        self.sample_sz = sample_size
        self.n_iter = num_iterations
        self.threshold = importance_threshold
        self.log_file = log_file or f"training_log_{model_name}.txt"

        os.makedirs(self.out_dir, exist_ok=True)
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        adata = ad.read_h5ad(self.data_path)
        self.feature_names = np.array(adata.var_names)
        # binary response: disease=1, healthy=0
        self.response = np.where(adata.obs["group"] == "disease", 1, 0)
        # balance by downsampling majority
        zeros, ones = np.bincount(self.response)
        if ones > zeros:
            idx0 = np.where(self.response == 0)[0]
            idx1 = np.where(self.response == 1)[0]
            np.random.seed(42)
            idx1_ds = np.random.choice(idx1, size=zeros, replace=False)
            sel = np.concatenate([idx0, idx1_ds])
        else:
            sel = np.arange(len(self.response))
        self.X_bal = adata.X[sel]
        self.y_bal = self.response[sel]

    def sample(self, iteration):
        np.random.seed(42 + iteration)
        idx0 = np.where(self.y_bal == 0)[0]
        idx1 = np.where(self.y_bal == 1)[0]
        if len(idx0) >= self.sample_sz and len(idx1) >= self.sample_sz:
            s0 = np.random.choice(idx0, size=self.sample_sz, replace=False)
            s1 = np.random.choice(idx1, size=self.sample_sz, replace=False)
            idx = np.concatenate([s0, s1])
        else:
            idx = np.arange(len(self.y_bal))
        return self.X_bal[idx], self.y_bal[idx]

    def select_features(self, fitted, X_test):
        # tree based classifier compute feature importances array after fitting
        if hasattr(fitted, "feature_importances_"):
            imp = fitted.feature_importances_
        # if coeff is returned
        elif hasattr(fitted, "coef_"):
            # permutation importance for coefficients
            perm = permutation_importance(
                fitted, X_test, self.y_test, n_repeats=3, random_state=42
            )
            imp = perm.importances_mean
        else:
            raise ValueError(
                "Model has neither feature_importances_ nor coef_"
            )
        return self.feature_names[imp > self.threshold].tolist()

    def run(self):
        selected_list = []
        accs = []

        for i in range(self.n_iter):
            X_s, y_s = self.sample(i)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_s, y_s, test_size=0.2, random_state=i
            )
            self.y_test = y_te  # used if we need permutation_importance

            logging.info(f"Iteration {i+1}: Training {self.name}")
            fitted = self.model.fit(X_tr, y_tr)

            # get selected features
            feats = self.select_features(fitted, X_te)
            selected_list.append(feats)
            # log the actual genes selected this round
            logging.info(f"Iteration {i+1}: Selected genes = {feats}")

            # evaluate
            preds = fitted.predict(X_te)
            acc = accuracy_score(y_te, preds)
            accs.append(acc)
            logging.info(f"Iteration {i+1}: Accuracy = {acc:.3f}")

        # tally frequencies
        freq = {}
        for feat_list in selected_list:
            for f in feat_list:
                freq[f] = freq.get(f, 0) + 1
        freq_df = pd.DataFrame.from_dict(
            freq, orient="index", columns=["frequency"]
        )
        freq_df.sort_values("frequency", ascending=False, inplace=True)

        # plots
        self.plot_histogram(freq_df)
        self.plot_bar(freq_df)

    def plot_histogram(self, freq_df):
        plt.figure(figsize=(10, 6))
        bins = np.arange(0, self.n_iter + 2) - 0.5
        plt.hist(freq_df["frequency"], bins=bins, edgecolor="black")
        plt.title(f"{self.name}: Gene selection frequency")
        plt.xlabel(f"Times selected (out of {self.n_iter})")
        plt.ylabel("Number of genes")
        fname = os.path.join(self.out_dir, f"{self.name}_hist.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved histogram to {fname}")

    def plot_bar(self, freq_df, cutoff: int = 5):
        plt.figure(figsize=(30, 6))
        df2 = freq_df[freq_df["frequency"] >= cutoff]
        gene_counts = df2["frequency"].to_dict()
        plt.bar(df2.index, df2["frequency"], edgecolor="black")
        plt.title(f"{self.name}: Genes selected ≥ {cutoff} times")
        plt.xticks(rotation=90)
        plt.ylabel("Frequency")
        fname = os.path.join(self.out_dir, f"{self.name}_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved barplot to {fname}")

        txt_path = os.path.join(
            self.out_dir, f"{self.name}_genes_over_{cutoff}.txt"
        )
        with open(txt_path, "w") as f:
            f.write("gene\tcount\n")
            for gene, count in gene_counts.items():
                f.write(f"{gene}\t{count}\n")
        print(f"Saved selected genes list to {txt_path}")
