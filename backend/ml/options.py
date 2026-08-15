"""ML options — the parameter definitions shown on the Prepare page.

Ported verbatim from CLASSify-2's api.py /get-ml-options and
/get-ml-options-uns endpoints.  One source of truth used by both
the /api/ml-options endpoint and the frontend.
"""

from __future__ import annotations

from typing import Any

SUPERVISED_MODELS = [
    "randomforest",
    "neuralnetwork",
    "tabpfn",
    "xgboost",
    "gradientboosting",
    "histgradientboosting",
    "bagging",
    "sgdclassifier",
    "logisticregression",
    "kneighbors",
]

UNSUPERVISED_MODELS = ["spectralclustering", "kmeans", "hdbscan"]

CLUSTERING_MODELS = {"spectralclustering": 1, "kmeans": 1, "hdbscan": 1}


def get_supervised_options() -> dict[str, Any]:
    """Return the supervised ML options dict (mirrors /get-ml-options)."""
    all_models = SUPERVISED_MODELS + UNSUPERVISED_MODELS
    return {
        "separate_testset": {
            "type": "bool",
            "default": False,
            "models": all_models,
            "help": "Upload a separate test set for model evaluation",
        },
        "shap_feature_explainability": {
            "type": "bool",
            "default": True,
            "models": SUPERVISED_MODELS,
            "help": "SHAP (SHapley Additive exPlanations) feature importance",
        },
        "synthesize_original": {
            "type": "bool",
            "default": False,
            "models": all_models + ["synthesis"],
            "help": "Generate synthetic data to balance the original training set",
        },
        "parameter_tune": {
            "type": "bool",
            "default": True,
            "models": all_models,
            "help": "Perform hyperparameter tuning using Optuna",
        },
        "synthesize_new": {
            "type": "bool",
            "default": False,
            "models": all_models + ["synthesis"],
            "help": "Generate an entirely new synthetic dataset",
        },
        "visualize": {
            "type": "bool",
            "default": True,
            "models": all_models,
            "help": "Display visualizations",
        },
        "standard_scaler": {
            "type": "bool",
            "default": False,
            "models": all_models,
            "help": "Use StandardScaler instead of MinMaxScaler (recommended for normally distributed data)",
        },
        "train_group": {
            "type": "list",
            "default": SUPERVISED_MODELS,
            "models": SUPERVISED_MODELS,
            "help": "Models to train",
        },
        "test_size": {
            "type": "float",
            "default": 0.2,
            "models": all_models,
            "help": "Proportion of data held out as test set",
        },
        "n_iter": {
            "type": "int",
            "default": 100,
            "models": all_models + ["parameter"],
            "help": "Number of parameter settings sampled during tuning",
        },
        "train_sample_type": {
            "type": "int",
            "default": 0,
            "models": all_models,
            "help": "Sampling strategy: 0=None, 1=Undersample, 2=Oversample",
        },
        "shap_diagram_features": {
            "type": "int",
            "default": 10,
            "models": SUPERVISED_MODELS,
            "help": "Maximum number of features displayed in SHAP diagrams",
        },
        "random_state": {
            "type": "int",
            "default": 42,
            "models": all_models,
            "help": "Random seed for reproducibility",
        },
        "shap_sample_size": {
            "type": "int",
            "default": 10,
            "models": SUPERVISED_MODELS,
            "help": "Number of rows to sample for SHAP per-row explanations",
        },
        "folds": {
            "type": "int",
            "default": 5,
            "models": all_models,
            "help": "Number of cross-validation folds",
        },
        "repeats": {
            "type": "int",
            "default": 1,
            "models": all_models,
            "help": "Number of cross-validation repeats",
        },
        "synthesize_model": {
            "type": "list",
            "default": ["tabular", "ctgan", "copulagan", "tvae"],
            "models": ["synthesis"],
            "help": "Synthetic data generation model (requires SDV addon)",
        },
        "parameter_goal": {
            "type": "list",
            "default": ["f1_macro", "precision_macro", "accuracy", "recall_macro"],
            "models": all_models + ["parameter"],
            "help": "Metric to optimize during parameter tuning",
        },
        "max_features": {
            "type": "list",
            "default": ["auto", "sqrt"],
            "models": ["randomforest"],
            "help": "Number of features to consider for best split",
        },
        "min_samples_split": {
            "type": "list",
            "default": [2, 5, 10],
            "models": ["randomforest"],
            "help": "Minimum samples required to split an internal node",
        },
        "min_samples_leaf": {
            "type": "list",
            "default": [1, 2, 4],
            "models": ["randomforest"],
            "help": "Minimum samples at a leaf node",
        },
        "bootstrap": {
            "type": "list",
            "default": [True, False],
            "models": ["randomforest"],
            "help": "Whether bootstrap samples are used for tree building",
        },
        "n_estimators_start": {
            "type": "int",
            "default": 10,
            "models": ["randomforest", "xgboost", "parameter"],
            "help": "Start: number of trees / boosting rounds",
        },
        "n_estimators_stop": {
            "type": "int",
            "default": 200,
            "models": ["randomforest", "xgboost", "parameter"],
            "help": "Stop: number of trees / boosting rounds",
        },
        "n_estimators_step": {
            "type": "int",
            "default": 10,
            "models": ["randomforest", "xgboost", "parameter"],
            "help": "Step size for number of trees",
        },
        "c_start": {
            "type": "float",
            "default": 0.1,
            "models": ["logisticregression", "parameter"],
            "help": "Start: regularization strength (inverse)",
        },
        "c_stop": {
            "type": "float",
            "default": 100,
            "models": ["logisticregression", "parameter"],
            "help": "Stop: regularization strength (inverse)",
        },
        "c_step": {
            "type": "float",
            "default": 0.1,
            "models": ["logisticregression", "parameter"],
            "help": "Step for regularization strength",
        },
        "max_depth_start": {
            "type": "int",
            "default": 5,
            "models": ["randomforest", "gradientboosting", "xgboost", "parameter"],
            "help": "Start: maximum tree depth",
        },
        "max_depth_stop": {
            "type": "int",
            "default": 500,
            "models": ["randomforest", "gradientboosting", "xgboost", "parameter"],
            "help": "Stop: maximum tree depth",
        },
        "max_depth_step": {
            "type": "int",
            "default": 10,
            "models": ["randomforest", "gradientboosting", "xgboost", "parameter"],
            "help": "Step for maximum tree depth",
        },
        "max_features_start": {
            "type": "float",
            "default": 0.02,
            "models": ["parameter"],
            "help": "Start: fraction of features for splits",
        },
        "max_features_stop": {
            "type": "float",
            "default": 1.0,
            "models": ["parameter"],
            "help": "Stop: fraction of features for splits",
        },
        "max_features_step": {
            "type": "float",
            "default": 0.02,
            "models": ["parameter"],
            "help": "Step for fraction of features",
        },
        "subsample_start": {
            "type": "float",
            "default": 0.1,
            "models": ["gradientboosting", "histgradientboosting", "xgboost", "parameter"],
            "help": "Start: fraction of samples for fitting trees",
        },
        "subsample_stop": {
            "type": "float",
            "default": 1.0,
            "models": ["gradientboosting", "histgradientboosting", "xgboost", "parameter"],
            "help": "Stop: fraction of samples for fitting trees",
        },
        "subsample_step": {
            "type": "float",
            "default": 0.1,
            "models": ["gradientboosting", "histgradientboosting", "xgboost", "parameter"],
            "help": "Step for fraction of samples",
        },
        "validation_fraction_start": {
            "type": "float",
            "default": 0.01,
            "models": ["gradientboosting", "histgradientboosting", "sgdclassifier", "parameter"],
            "help": "Start: proportion of training data for validation",
        },
        "validation_fraction_stop": {
            "type": "float",
            "default": 0.5,
            "models": ["gradientboosting", "histgradientboosting", "sgdclassifier", "parameter"],
            "help": "Stop: proportion of training data for validation",
        },
        "validation_fraction_step": {
            "type": "float",
            "default": 0.01,
            "models": ["gradientboosting", "histgradientboosting", "sgdclassifier", "parameter"],
            "help": "Step for validation fraction",
        },
        "n_iter_no_change_start": {
            "type": "int",
            "default": 5,
            "models": ["gradientboosting", "histgradientboosting", "sgdclassifier", "parameter"],
            "help": "Start: iterations without improvement before early stopping",
        },
        "n_iter_no_change_stop": {
            "type": "int",
            "default": 50,
            "models": ["gradientboosting", "histgradientboosting", "sgdclassifier", "parameter"],
            "help": "Stop: iterations without improvement before early stopping",
        },
        "n_iter_no_change_step": {
            "type": "int",
            "default": 5,
            "models": ["gradientboosting", "histgradientboosting", "sgdclassifier", "parameter"],
            "help": "Step for early stopping iterations",
        },
        "learning_rate_start": {
            "type": "float",
            "default": 0.01,
            "models": ["histgradientboosting", "parameter"],
            "help": "Start: learning rate for gradient boosting",
        },
        "learning_rate_stop": {
            "type": "float",
            "default": 1.0,
            "models": ["histgradientboosting", "parameter"],
            "help": "Stop: learning rate for gradient boosting",
        },
        "learning_rate_step": {
            "type": "float",
            "default": 0.01,
            "models": ["histgradientboosting", "parameter"],
            "help": "Step for learning rate",
        },
        "nn_hidden_layer_sizes_start": {
            "type": "int",
            "default": 10,
            "models": ["neuralnetwork", "parameter"],
            "help": "Start: nodes per hidden layer",
        },
        "nn_hidden_layer_sizes_stop": {
            "type": "int",
            "default": 1000,
            "models": ["neuralnetwork", "parameter"],
            "help": "Stop: nodes per hidden layer",
        },
        "nn_hidden_layer_sizes_step": {
            "type": "int",
            "default": 10,
            "models": ["neuralnetwork", "parameter"],
            "help": "Step for hidden layer sizes",
        },
        "nn_hidden_layer_depth_start": {
            "type": "int",
            "default": 1,
            "models": ["neuralnetwork", "parameter"],
            "help": "Start: number of hidden layers",
        },
        "nn_hidden_layer_depth_stop": {
            "type": "int",
            "default": 2,
            "models": ["neuralnetwork", "parameter"],
            "help": "Stop: number of hidden layers",
        },
        "nn_hidden_layer_depth_step": {
            "type": "int",
            "default": 1,
            "models": ["neuralnetwork", "parameter"],
            "help": "Step for number of hidden layers",
        },
        "nn_learning_rate_start": {
            "type": "float",
            "default": 0.0001,
            "models": ["neuralnetwork", "parameter"],
            "help": "Start: NN learning rate",
        },
        "nn_learning_rate_stop": {
            "type": "float",
            "default": 0.01,
            "models": ["neuralnetwork", "parameter"],
            "help": "Stop: NN learning rate",
        },
        "nn_learning_rate_step": {
            "type": "float",
            "default": 0.001,
            "models": ["neuralnetwork", "parameter"],
            "help": "Step for NN learning rate",
        },
        "alpha_start": {
            "type": "float",
            "default": 0.000001,
            "models": ["neuralnetwork", "sgdclassifier", "parameter"],
            "help": "Start: regularization penalty",
        },
        "alpha_stop": {
            "type": "float",
            "default": 0.001,
            "models": ["neuralnetwork", "sgdclassifier", "parameter"],
            "help": "Stop: regularization penalty",
        },
        "alpha_step": {
            "type": "float",
            "default": 0.00001,
            "models": ["neuralnetwork", "sgdclassifier", "parameter"],
            "help": "Step for regularization penalty",
        },
    }


def get_unsupervised_options() -> dict[str, Any]:
    """Return the unsupervised ML options dict (mirrors /get-ml-options-uns)."""
    return {
        "parameter_tune": {
            "type": "bool",
            "default": True,
            "models": UNSUPERVISED_MODELS,
            "help": "Perform hyperparameter tuning",
        },
        "visualize": {
            "type": "bool",
            "default": True,
            "models": UNSUPERVISED_MODELS,
            "help": "Display visualizations",
        },
        "standard_scaler": {
            "type": "bool",
            "default": False,
            "models": UNSUPERVISED_MODELS,
            "help": "Use StandardScaler instead of MinMaxScaler",
        },
        "train_group": {
            "type": "list",
            "default": UNSUPERVISED_MODELS,
            "models": UNSUPERVISED_MODELS,
            "help": "Models to train",
        },
        "num_clusters": {
            "type": "int",
            "default": 2,
            "models": ["spectralclustering", "kmeans"],
            "help": "Number of clusters (if Parameter Tune is unchecked)",
        },
        "n_iter": {
            "type": "int",
            "default": 100,
            "models": UNSUPERVISED_MODELS + ["parameter"],
            "help": "Number of parameter settings sampled during tuning",
        },
        "random_state": {
            "type": "int",
            "default": 42,
            "models": UNSUPERVISED_MODELS,
            "help": "Random seed for reproducibility",
        },
        "clustering_parameter_goal": {
            "type": "list",
            "default": ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"],
            "models": UNSUPERVISED_MODELS + ["parameter"],
            "help": "Metric to optimize during parameter tuning",
        },
        "num_clusters_start": {
            "type": "int",
            "default": 2,
            "models": ["spectralclustering", "kmeans", "parameter"],
            "help": "Start: number of clusters",
        },
        "num_clusters_stop": {
            "type": "int",
            "default": 5,
            "models": ["spectralclustering", "kmeans", "parameter"],
            "help": "Stop: number of clusters",
        },
        "num_clusters_step": {
            "type": "int",
            "default": 1,
            "models": ["spectralclustering", "kmeans", "parameter"],
            "help": "Step for number of clusters",
        },
        "min_cluster_size_start": {
            "type": "int",
            "default": 5,
            "models": ["hdbscan", "parameter"],
            "help": "Start: minimum points per cluster",
        },
        "min_cluster_size_stop": {
            "type": "int",
            "default": 100,
            "models": ["hdbscan", "parameter"],
            "help": "Stop: minimum points per cluster",
        },
        "min_cluster_size_step": {
            "type": "int",
            "default": 5,
            "models": ["hdbscan", "parameter"],
            "help": "Step for minimum cluster size",
        },
        "min_samples_start": {
            "type": "int",
            "default": 5,
            "models": ["hdbscan", "parameter"],
            "help": "Start: minimum samples (higher = more noise)",
        },
        "min_samples_stop": {
            "type": "int",
            "default": 200,
            "models": ["hdbscan", "parameter"],
            "help": "Stop: minimum samples",
        },
        "min_samples_step": {
            "type": "int",
            "default": 2,
            "models": ["hdbscan", "parameter"],
            "help": "Step for minimum samples",
        },
        "cluster_selection_epsilon_start": {
            "type": "float",
            "default": 0.0,
            "models": ["hdbscan", "parameter"],
            "help": "Start: cluster merge distance threshold",
        },
        "cluster_selection_epsilon_stop": {
            "type": "float",
            "default": 1.0,
            "models": ["hdbscan", "parameter"],
            "help": "Stop: cluster merge distance threshold",
        },
        "cluster_selection_epsilon_step": {
            "type": "float",
            "default": 0.1,
            "models": ["hdbscan", "parameter"],
            "help": "Step for cluster selection epsilon",
        },
        "nearest_neighbors_start": {
            "type": "int",
            "default": 5,
            "models": ["spectralclustering", "parameter"],
            "help": "Start: number of nearest neighbors",
        },
        "nearest_neighbors_stop": {
            "type": "int",
            "default": 50,
            "models": ["spectralclustering", "parameter"],
            "help": "Stop: number of nearest neighbors",
        },
        "nearest_neighbors_step": {
            "type": "int",
            "default": 5,
            "models": ["spectralclustering", "parameter"],
            "help": "Step for number of nearest neighbors",
        },
        "num_components_start": {
            "type": "int",
            "default": 2,
            "models": ["spectralclustering", "parameter"],
            "help": "Start: number of eigenvectors for spectral embedding",
        },
        "num_components_stop": {
            "type": "int",
            "default": 15,
            "models": ["spectralclustering", "parameter"],
            "help": "Stop: number of eigenvectors for spectral embedding",
        },
        "num_components_step": {
            "type": "int",
            "default": 2,
            "models": ["spectralclustering", "parameter"],
            "help": "Step for number of eigenvectors",
        },
        "max_iter_start": {
            "type": "int",
            "default": 100,
            "models": ["kmeans", "parameter"],
            "help": "Start: maximum iterations for cluster updates",
        },
        "max_iter_stop": {
            "type": "int",
            "default": 500,
            "models": ["kmeans", "parameter"],
            "help": "Stop: maximum iterations for cluster updates",
        },
        "max_iter_step": {
            "type": "int",
            "default": 50,
            "models": ["kmeans", "parameter"],
            "help": "Step for maximum iterations",
        },
        "n_init_start": {
            "type": "int",
            "default": 5,
            "models": ["kmeans", "parameter"],
            "help": "Start: number of random initializations",
        },
        "n_init_stop": {
            "type": "int",
            "default": 20,
            "models": ["kmeans", "parameter"],
            "help": "Stop: number of random initializations",
        },
        "n_init_step": {
            "type": "int",
            "default": 5,
            "models": ["kmeans", "parameter"],
            "help": "Step for number of initializations",
        },
        "affinity_method": {
            "type": "list",
            "default": ["rbf", "nearest_neighbors"],
            "models": ["spectralclustering", "parameter"],
            "help": "Similarity matrix calculation method",
        },
    }


def get_options(supervised: bool = True) -> dict[str, Any]:
    """Return the appropriate options dict based on the training mode."""
    if supervised:
        return get_supervised_options()
    return get_unsupervised_options()
