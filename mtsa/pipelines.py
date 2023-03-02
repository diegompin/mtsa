from sklearn import metrics
from mtsa.utils import (
    files_train_test_split
)
from mtsa import (
    FEATURES,
    FINAL_MODEL,
    MFCCMix, 
    MFCCMixCV
)


from sklearn.mixture import GaussianMixture
import glob
import os
from functools import reduce
import numpy as np

from sklearn.manifold import TSNE
class MFCCMixPipeline():

    def __init__(
        self, 
        # final_model=GaussianMixture,
        # features=FEATURES,
        # sampling_rate=16000,
        *args, 
        **kwargs) -> None:
        self.__dict__.update(kwargs)
        # self.final_model = FINAL_MODEL
        
        final_model = GaussianMixture(n_components=self.n_components)
        self.mtsa = MFCCMix(
            sampling_rate=self.sampling_rate,   
            features=self.features,
            final_model=final_model
        )

        # self.atelier = AtelierCV(
        #     final_model=final_model,
        #     sampling_rate=self.sampling_rate   
        # )

    def calculate_aucroc(self, X_train, X_test, y_train, y_test):
        self.mtsa.fit(X_train, y_train)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, self.tsa.score_samples(X_test))
        auc = metrics.auc(fpr, tpr)
        return auc
    
    
    def plot_tsne(self, X_train, X_test, y_train, y_test, **kwargs):
        self.mtsa.fit(X_train, y_train)
        
        perplexity = 4
        if 'perplexity' in kwargs:
            perplexity = kwargs['perplexity']
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300, random_state=None)
        
        X = self.mtsa.model['features'].fit_transform(
            self.mtsa.model['array2mfcc'].fit_transform(
            self.mtsa.model['wav2array'].fit_transform(X_test)))
        
        tsne_results = tsne.fit_transform(X)
        x = tsne_results[:,0]
        y = tsne_results[:,1]
        hue = y_test
        return x, y, hue
        

    def plot_tsne_individual(self, filepath, **kwargs):
        X_train, X_test, y_train, y_test = files_train_test_split(filepath)
        x, y, hue = self.plot_tsne(X_train, X_test, y_train, y_test, **kwargs)
        return x, y, hue


    # def calculate_aucroc_individual(self, filepath):
    #     X_train, X_test, y_train, y_test = files_train_test_split(filepath)
    #     roc = self.calculate_aucroc(X_train, X_test, y_train, y_test)
    #     return roc

    def calculate_aucroc_combined(self, filepath):
        def reduce_data(d1, d2):
            data = zip(d1, d2)
            data = list(data)
            X_train, X_test, y_train, y_test = [np.concatenate([d1, d2]) for (d1,d2) in data]
            return X_train, X_test, y_train, y_test
        paths = glob.glob(os.path.join(filepath, f'*{os.sep}' ))
        data = list(map(files_train_test_split, paths))
        X_train, X_test, y_train, y_test = reduce(reduce_data, data)
        roc = self.calculate_aucroc(X_train, X_test, y_train, y_test)
        return roc
   