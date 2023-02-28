
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np
from mtsa.utils import (
    AbnormalSplit,
    Demux2Array,
    Wav2Array,
    files_train_test_split
)
from mtsa.features.mel import (
    Array2MelSpec
) 
from sklearn.pipeline import (
    Pipeline, 
)
from functools import reduce

class AutoEncoderMixin(Model):
            def score_samples(self, X):
                return -1 * np.mean(np.square(X - self.predict(X))) 
            
            def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=0, callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
                return super().fit(x, x, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
            
       
class Hitachi(BaseEstimator, OutlierMixin):
    """
    
      :References:
    Purohit, Harsh, et al. "MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection." arXiv preprint arXiv:1909.09347 (2019).
    
    """
    def __init__(self, 
                 sampling_rate=None,
                 random_state = None,
                 n_mels=64,
                 frames=5,
                 n_fft=1024,
                 hop_length=512,
                 power=2.0,
                 mono=False,
                 epochs=50,
                 batch_size=512,
                 shuffle=True,
                 validation_split=0.1,
                 verbose=0
                 ) -> None:
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        self.n_mels=n_mels
        self.frames=frames
        self.n_fft=n_fft
        self.hop_length=hop_length
        self.power=power
        self.mono = mono
        self.epochs=epochs
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.validation_split=validation_split
        self.verbose=verbose
        self.model = self._build_model()
    
    def fit(self, X, y=None):
        return self.model.fit(X, 
                              y,
                              final_model__batch_size=self.batch_size,
                              final_model__shuffle= self.shuffle,
                              final_model__validation_split=self.validation_split,
                              final_model__verbose=self.verbose,
                              )

    def transform(self, X, y=None):

        def transform_one(Xi):
            l = list()
            l.append(Xi)
            l.extend(self.model.steps[:-1])
            Xt = reduce(lambda x, y: y[1].transform(x), l)
            return Xt



        Xp = np.array(
            list(
                map(
                    self.predict, 
                    [[x] for x in X])
                )
            )


        Xt = np.array(
            list(
                map(
                    transform_one, 
                    [[x] for x in X])
                )
            )

        Xf = np.mean(Xt - Xp, axis=1)
        return Xf

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return np.array(
            list(
                map(
                    self.model.score_samples, 
                    [[x] for x in X])
                )
            )
  
    
    def get_model(self):
        inputDim = self.n_mels * self.frames
        inputLayer = Input(shape=(inputDim,))
        h = Dense(64, activation="relu")(inputLayer)
        h = Dense(64, activation="relu")(h)
        h = Dense(8, activation="relu")(h)
        h = Dense(64, activation="relu")(h)
        h = Dense(64, activation="relu")(h)
        h = Dense(inputDim, activation=None)(h)
        final_model = AutoEncoderMixin(inputs=inputLayer, outputs=h)
        final_model.compile(optimizer='adam', loss='mean_squared_error')
        return final_model
    
    def _build_model(self):
        wav2array = Wav2Array(
            sampling_rate=self.sampling_rate,
            mono=self.mono,
            )
        
        demux2array = Demux2Array()
        array2melspec= Array2MelSpec(
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            frames=self.frames,
            power=self.power,
            )
        
        final_model = self.get_model()
        
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("demux2array", demux2array),
                ("array2melspec", array2melspec),
                ("final_model", final_model),
                ]
            )
        
        return model


class HitachiDCASE2020(Hitachi):
    

    def get_model(self):
        inputDim = self.n_mels * self.frames
        inputLayer = Input(shape=(inputDim,))
        h = Dense(128, activation="relu")(inputLayer)
        h = Dense(128, activation="relu")(h)
        h = Dense(128, activation="relu")(h)
        h = Dense(128, activation="relu")(h)
        h = Dense(8, activation="relu")(h)
        h = Dense(128, activation="relu")(inputLayer)
        h = Dense(128, activation="relu")(h)
        h = Dense(128, activation="relu")(h)
        h = Dense(128, activation="relu")(h)
        h = Dense(inputDim, activation=None)(h)
        final_model = AutoEncoderMixin(inputs=inputLayer, outputs=h)
        final_model.compile(optimizer='adam', loss='mean_squared_error')
        return final_model