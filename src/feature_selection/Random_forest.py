from common import RobustFeatureSelector
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_split=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
selector = RobustFeatureSelector(
    model=model,
    model_name="random_forest",
    data_path="data/adata_balanced.h5ad",
    output_folder="results/rf",
    log_file="training_log_rf.txt",
    importance_threshold=0.001,
)
selector.run()
