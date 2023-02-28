import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.pipeline import (
    Pipeline, 
    FeatureUnion
) 
from mtsa.features.stats import (
    MagnitudeMeanFeatureMfcc, 
    MagnitudeStdFeatureMfcc, 
    CorrelationFeatureMfcc
    )

from mtsa.features.mel import (
    Array2Mfcc 
)
from mtsa.utils import (
    Wav2Array,
)

from sklearn.mixture import GaussianMixture
from functools import reduce

FEATURES = [
      ("M", MagnitudeMeanFeatureMfcc()), 
      ("S", MagnitudeStdFeatureMfcc()), 
      ("C", CorrelationFeatureMfcc())
]

FINAL_MODEL = GaussianMixture()

class MFCCMix(BaseEstimator, OutlierMixin):

    def __init__(self, 
                 final_model=FINAL_MODEL, 
                 features=FEATURES,
                 sampling_rate=None,
                 random_state = None,
                 ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.final_model = final_model
        self.random_state = random_state
        self.features = features
        self.model = self._build_model()
        
    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.score_samples(X=X)

    def _build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        features = FeatureUnion(self.features)
        
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("array2mfcc", array2mfcc),
                ("features", features),
                ("final_model", self.final_model),
                ]
            )
        
        return model

