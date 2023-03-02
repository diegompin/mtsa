from mtsa import MFCCMix

from sklearn.model_selection import (
    RandomizedSearchCV,
    ShuffleSplit
)

from sklearn.metrics import (
    make_scorer,
    recall_score
)

import numpy as np


class MFCCMixCV(MFCCMix):
    
    def __init__(
        self,
        sampling_rate = 16000,
        random_state = None,
        cv=None) -> None:
        super(MFCCMixCV, self).__init__(
            random_state=random_state,
            sampling_rate=sampling_rate
            )
        if not cv:
            cv = ShuffleSplit(
                random_state=random_state,
                n_splits=5)
            self.cv = cv
        
        params = [
                {
                    "final_model__covariance_type": ['full', 'tied', 'diag', 'spherical'],
                    # "final_model__n_components": [n for n in np.arange(1, 230)]
                    # "final_model__covariance_type": ['full'],
                    "final_model__n_components": np.arange(230, 231)
                },
        ]
        
        self.params = params

        def my_score(y_true, y_pred):
            y_pred = 1*(y_pred == 1)
            return recall_score(y_true=y_true, y_pred=y_pred)
        my_scorer = make_scorer(my_score)
        search = RandomizedSearchCV(
            n_iter=4,
            estimator=self.model,
            param_distributions=self.params,
            scoring=my_scorer,
            # scoring=my_score,
            cv=self.cv
        )
        self.search = search

    def fit(self, X, y=None, **fit_params):
        return self.search.fit(X, y)

    def transform(self, X, y=None, **fit_params):
        return self.search.transform(X, y, **fit_params)

    def predict(self, X):
        return self.search.predict(X)

    def score_samples(self, X):
        return self.search.score_samples(X=X)