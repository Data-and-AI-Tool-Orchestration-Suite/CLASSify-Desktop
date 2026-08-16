"""Typed training arguments — replaces the dotdict from CLASSify-2's utils.py.

Every parameter from the web app's /get-ml-options endpoint is represented
here with its default value.  The engine receives a fully-populated
TrainingArgs instance instead of a loosely-typed dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingArgs:
    """All parameters for a single training job.

    Defaults match the web app's /get-ml-options endpoint exactly.
    Fields marked ``# set by runner`` are populated by the job runner,
    not by the user-facing options form.
    """

    # ── User-facing options (from the Prepare page) ──
    supervised: bool = True
    autodetermineclusters: bool = False

    separate_testset: bool = False
    shap_feature_explainability: bool = True
    synthesize_original: bool = False
    parameter_tune: bool = True
    synthesize_new: bool = False
    visualize: bool = True
    standard_scaler: bool = False

    train_group: list[str] = field(
        default_factory=lambda: [
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
            "spectralclustering",
            "kmeans",
        ]
    )
    test_size: float = 0.2
    n_iter: int = 100
    train_sample_type: int = 0  # 0=no sampling, 1=undersample, 2=oversample
    shap_diagram_features: int = 10
    random_state: int = 42
    shap_sample_size: int = 10
    folds: int = 5
    repeats: int = 1

    synthesize_model: list[str] = field(
        default_factory=lambda: ["tabular", "ctgan", "copulagan", "tvae"]
    )
    parameter_goal: list[str] = field(
        default_factory=lambda: ["f1_macro", "precision_macro", "accuracy", "recall_macro"]
    )

    # Model parameter ranges (start/stop/step)
    n_estimators_start: int = 10
    n_estimators_stop: int = 200
    n_estimators_step: int = 10
    c_start: float = 0.1
    c_stop: float = 100
    c_step: float = 0.1
    max_depth_start: int = 5
    max_depth_stop: int = 500
    max_depth_step: int = 10
    max_features_start: float = 0.02
    max_features_stop: float = 1.0
    max_features_step: float = 0.02
    subsample_start: float = 0.1
    subsample_stop: float = 1.0
    subsample_step: float = 0.1
    validation_fraction_start: float = 0.01
    validation_fraction_stop: float = 0.5
    validation_fraction_step: float = 0.01
    n_iter_no_change_start: int = 5
    n_iter_no_change_stop: int = 50
    n_iter_no_change_step: int = 5
    learning_rate_start: float = 0.01
    learning_rate_stop: float = 1.0
    learning_rate_step: float = 0.01
    nn_hidden_layer_sizes_start: int = 10
    nn_hidden_layer_sizes_stop: int = 1000
    nn_hidden_layer_sizes_step: int = 10
    nn_hidden_layer_depth_start: int = 1
    nn_hidden_layer_depth_stop: int = 2
    nn_hidden_layer_depth_step: int = 1
    nn_learning_rate_start: float = 0.0001
    nn_learning_rate_stop: float = 0.01
    nn_learning_rate_step: float = 0.001
    alpha_start: float = 0.000001
    alpha_stop: float = 0.001
    alpha_step: float = 0.00001

    # Unsupervised params
    num_clusters: int = 2
    clustering_parameter_goal: list[str] = field(
        default_factory=lambda: [
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
        ]
    )
    num_clusters_start: int = 2
    num_clusters_stop: int = 5
    num_clusters_step: int = 1
    min_cluster_size_start: int = 5
    min_cluster_size_stop: int = 100
    min_cluster_size_step: int = 5
    min_samples_start: int = 5
    min_samples_stop: int = 200
    min_samples_step: int = 2
    cluster_selection_epsilon_start: float = 0.0
    cluster_selection_epsilon_stop: float = 1.0
    cluster_selection_epsilon_step: float = 0.1
    nearest_neighbors_start: int = 5
    nearest_neighbors_stop: int = 50
    nearest_neighbors_step: int = 5
    num_components_start: int = 2
    num_components_stop: int = 15
    num_components_step: int = 2
    max_iter_start: int = 100
    max_iter_stop: int = 500
    max_iter_step: int = 50
    n_init_start: int = 5
    n_init_stop: int = 20
    n_init_step: int = 5
    affinity_method: list[str] = field(default_factory=lambda: ["rbf", "nearest_neighbors"])

    # Fixed arrays used internally by the engine
    max_features: list[str] = field(default_factory=lambda: ["sqrt", "log2"])
    min_samples_split: list[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: list[int] = field(default_factory=lambda: [1, 2, 4])
    bootstrap: list[bool] = field(default_factory=lambda: [True, False])

    # ── Set by the runner, not the user ──
    class_column: str | None = None
    report_uuid: str = ""
    file_directory: str = ""
    disable_model_save: bool = False
    n_jobs: int = 0  # 0 = os.cpu_count()
    multiclass: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingArgs:
        """Build TrainingArgs from a loosely-typed dict (e.g. from the API).

        Handles the web app's stringified bools ('True'/'False') and
        list-as-string representations.
        """
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}

        for key, value in data.items():
            if key not in field_names:
                continue
            if isinstance(value, str):
                if value == "True":
                    kwargs[key] = True
                elif value == "False":
                    kwargs[key] = False
                else:
                    try:
                        if "." in value or "e" in value.lower():
                            kwargs[key] = float(value)
                        else:
                            kwargs[key] = int(value)
                    except ValueError:
                        kwargs[key] = value
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict (for DB storage)."""
        from dataclasses import asdict

        return asdict(self)
