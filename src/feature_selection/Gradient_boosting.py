from common import RobustFeatureSelector
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    max_iter=1000,
    max_depth=6,
    learning_rate=0.1,
    min_samples_leaf=5,
    random_state=42,
    verbose=1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
)
selector = RobustFeatureSelector(
    model=model,
    model_name="gradient_boosting",
    data_path="data/adata_balanced.h5ad",
    output_folder="results/gb",
    log_file="training_log_gb.txt",
)
selector.run()
