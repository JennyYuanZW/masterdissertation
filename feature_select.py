import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import argparse
import joblib
import os
import gc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

parser = argparse.ArgumentParser(description="Feature Selection on HPC")
parser.add_argument(
    "--input_files",
    type=str,
    nargs="+",
    required=True,
    help="List of .h5ad files to concatenate",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Output directory for selected genes",
)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

print("Loading and concatenating data...")

adata_combined = None

for batch_id, file in enumerate(args.input_files):
    print(f"Loading file: {file}")

    try:
        adata = ad.read_h5ad(file, backed="r")
        print(f"Opened {file} with shape {adata.shape}")

        # Close backed file
        adata.file.close()
        del adata
        gc.collect()

        # Load fully into memory for modifications
        adata = ad.read_h5ad(file)
        adata.obs["batch"] = batch_id  # Mark batch ID

        # Concatenate to combined dataset
        if adata_combined is None:
            adata_combined = adata  # First dataset, initialize here
        else:
            adata_combined = ad.concat(
                [adata_combined, adata], join="inner", label="batch"
            )

        # Free memory from the current file
        del adata
        gc.collect()

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Check if concatenation was successful
if adata_combined is None:
    print("No valid .h5ad files loaded. Exiting.")
    exit(1)

print("Concatenation completed successfully!")
adata_subset = adata_combined[
    adata_combined.obs["predicted.id"] == "Microglia"
].copy()
# Batch effect correction
print("Applying Combat batch correction...")
try:
    sc.pp.combat(adata_combined, key="batch")
    print("Combat batch correction applied successfully.")
except Exception as e:
    print(f"Error in Combat batch correction: {e}")


# Remove genes with zero expression
print("Removing genes with zero expression across all cells...")
nonzero_genes = np.array(adata_subset.X.sum(axis=0)).flatten() > 0
adata_subset = adata_subset[:, nonzero_genes].copy()
print(f"Remaining genes after filtering: {adata_subset.shape[1]}")

adata_subset.X = np.nan_to_num(adata_subset.X)  # Handle NaN values
adata_subset.X = sp.csc_matrix(adata_subset.X)  # Convert to sparse format

# Standardize features
print("Standardizing gene expression data...")
try:
    scaler = StandardScaler(with_mean=False)
    subset_scaled = scaler.fit_transform(adata_subset.X)
    print("Standardization completed successfully.")
except Exception as e:
    print(f"Error during standardization: {e}")
    exit(1)

# Prepare response labels
print("Preparing response labels...")
try:
    response = np.where(adata_subset.obs["group"] == "disease", 1, 0)
    print(
        f"Response label counts: {np.bincount(response)}"
    )  # Debug: Check label distribution
except Exception as e:
    print(f"Error during response label preparation: {e}")
    exit(1)

# LASSO Feature Selection
print("Running LASSO feature selection...")
try:
    lasso = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        cv=5,
        Cs=np.logspace(
            -2, 2, 20
        ),  # Increase the range of C values (reduce penalty)
        verbose=1,
    ).fit(subset_scaled, response)

    nonzero_indices = np.where(lasso.coef_ != 0)[1]
    lasso_genes = adata_subset.var_names[nonzero_indices].tolist()

    lasso_output_path = os.path.join(
        args.output_dir, "lasso_selected_genes.txt"
    )
    with open(lasso_output_path, "w") as f:
        f.write("\n".join(lasso_genes))

    print(
        f"LASSO selected {len(lasso_genes)} genes. Saved to {lasso_output_path}"
    )
except Exception as e:
    print(f"Error in LASSO feature selection: {e}")

# Random Forest Feature Selection
print("Running Random Forest feature selection...")
try:
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    # Get feature importances
    feature_importances = rf.feature_importances_
    selected_features = np.array(adata_subset.var_names)

    # Convert to DataFrame for better readability
    importance_df = pd.DataFrame(
        {"Feature": selected_features, "Importance": feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Select only the top 10% most important features
    top_1_percent = int(
        len(feature_importances) * 0.01
    )  # Keep only the top 10% of features
    top_features = importance_df.head(top_1_percent)["Feature"].tolist()

    rf_output_path = os.path.join(args.output_dir, "rf_selected_genes.txt")
    with open(rf_output_path, "w") as f:
        f.write("\n".join(top_features))

    print(
        f"Random Forest selected {len(top_features)} genes. Saved to {rf_output_path}"
    )
except Exception as e:
    print(f"Error in Random Forest feature selection: {e}")

# GM Boost (Gradient Boosting) Feature Selection
print("Running Gradient Boosting (GM Boost) feature selection...")
try:
    print("Running Gradient Boosting (GM Boost) feature selection...")

    gb = HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    gb.fit(subset_scaled.toarray(), response)

    # Compute permutation importance
    perm_importance = permutation_importance(
        gb,
        subset_scaled.toarray(),
        response,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    feature_importances = perm_importance.importances_mean

    # Option 1: Select a fixed number of top features
    N = 500  # Adjust to the number of top features you want to keep
    sorted_indices = np.argsort(feature_importances)[
        ::-1
    ]  # Sort in descending order
    top_indices = sorted_indices[:N]  # Select top N features

    # Option 2: Select top X% of features
    top_percentage = 0.01  # Adjust percentage (e.g., 0.05 for top 5%)
    num_selected = int(len(feature_importances) * top_percentage)
    top_indices = sorted_indices[:num_selected]  # Select top X% features

    # Map selected indices back to gene names
    gb_genes = adata_subset.var_names[top_indices].tolist()

    # Save selected features
    gb_output_path = os.path.join(args.output_dir, "gb_selected_genes.txt")
    with open(gb_output_path, "w") as f:
        f.write("\n".join(gb_genes))

    print(
        f"GM Boost selected {len(gb_genes)} genes. Saved to {gb_output_path}"
    )

except Exception as e:
    print(f"Error in GM Boost feature selection: {e}")

print("Feature selection pipeline completed successfully!")
