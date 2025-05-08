from common import RobustFeatureSelector
from sklearn.linear_model import LogisticRegressionCV

lasso = LogisticRegressionCV(
    penalty="l1",
    solver="saga",
    cv=3,
    Cs=np.logspace(-3, -0.3, 10),
    max_iter=10000,
    tol=0.0005,
    verbose=1,
    random_state=42,
)
selector = RobustFeatureSelector(
    model=model,
    model_name="lasso_select",
    data_path="data/adata_balanced.h5ad",
    output_folder="results/lasso",
    log_file="training_log_lasso.txt",
)
selector.run()
