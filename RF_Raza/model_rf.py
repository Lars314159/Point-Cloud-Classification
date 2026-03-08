"""
Random Forest model factory for hyperspectral classification.
"""

from sklearn.ensemble import RandomForestClassifier


def create_random_forest(
    n_features: int,
    n_classes: int,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float = 'sqrt',
    class_weight: str | None = None,
    oob_score: bool = False,
    random_state: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=-1,
    )
