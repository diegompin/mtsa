import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Model, model_from_json
from joblib import dump, load
from scipy.signal import find_peaks
from tensorflow.python.keras.layers import Input
from mtsa.utils import Wav2Array
from spectrum import Periodogram
from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, OutlierMixin
from mtsa.models.ransyncoders_components.rancoders import RANCoders
from mtsa.models.ransyncoders_components.frequencycoder import FrequencyCoder
from mtsa.models.ransyncoders_components.sinusoidalcoder import SinusoidalCoder
from mtsa.models.ransyncoders_components.exceptions.parametererror import ParameterError

class RANSynCodersBase():
    def __init__(
            self,
            n_estimators: int,
            max_features: int,
            encoding_depth: int,
            latent_dim: int, 
            decoding_depth: int,
            activation: str,
            output_activation: str,
            delta: float,
            synchronize: bool, 
            force_synchronization: bool,
            min_periods: int,
            freq_init: Optional[List[float]],
            max_freqs: int, 
            min_dist: int,
            trainable_freq: bool,
            bias: bool,
            sampling_rate: int,
            mono: bool
    ):
        # Rancoders inputs:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.activation = activation
        self.output_activation = output_activation
        self.delta = delta
        
        # Syncrhonization inputs
        self.synchronize = synchronize
        self.force_synchronization = force_synchronization
        self.min_periods = min_periods
        self.freq_init = freq_init  # in radians (angular frequency)
        self.max_freqs = max_freqs
        self.min_dist = min_dist
        self.trainable_freq = trainable_freq
        self.bias = bias
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.min_max_scaler = MinMaxScaler()
        # set all variables to default to float32
        tf.keras.backend.set_floatx('float32')

    def build(self, input_shape, learning_rate: float, initial_stage: bool = False):
        x_in = Input(shape=(input_shape[-1],))  # created for either raw signal or synchronized signal
        optmizer = adam_v2.Adam(learning_rate = learning_rate)
        if initial_stage:
            freq_out = FrequencyCoder()(x_in)
            self.freqcoder = Model(inputs=x_in, outputs=freq_out)
            self.freqcoder.compile(optimizer=optmizer, loss=lambda y,f: quantile_loss(0.5, y,f))
        else:
            bounds_out = RANCoders(
                    n_estimators = self.n_estimators,
                    max_features = self.max_features,
                    encoding_depth = self.encoding_depth,
                    latent_dim = self.latent_dim,
                    decoding_depth = self.decoding_depth,
                    delta = self.delta,
                    activation = self.activation,
                    output_activation = self.output_activation,
                    name='rancoders'
                )(x_in)
            self.rancoders = Model(inputs=x_in, outputs=bounds_out)
            self.rancoders.compile(
                    optimizer=optmizer, 
                    loss=[lambda y,f: quantile_loss(1-self.delta, y,f), lambda y,f: quantile_loss(self.delta, y,f)]
            )  
            K.set_value(self.rancoders.optimizer.learning_rate, 0.01)
            if self.synchronize:
                t_in = Input(shape=(input_shape[-1],))
                sin_out = SinusoidalCoder(freq_init=self.freq_init, trainable_freq=self.trainable_freq)(t_in)
                self.sincoder = Model(inputs=t_in, outputs=sin_out)
                self.sincoder.compile(optimizer=optmizer, loss=lambda y,f: quantile_loss(0.5, y,f))
        
    def fit(
            self, 
            X: np.ndarray,
            y, 
            timestamps_matrix: np.ndarray,
            learning_rate: float,
            batch_size: int, #old default value = 360
            epochs: int, #old default value = 100 
            freq_warmup: int,  # number of warmup epochs to prefit the frequency = 10 (old default value)
            sin_warmup: int,  # number of warmup epochs to prefit the sinusoidal representation = 10 (old default value)
            pos_amp: bool,  # whether to constraint amplitudes to be +ve only
    ):
        self.batch_size = batch_size
        self.experiment_dataframe = self.__get_experiment_dataframe()

        datasets, X = self.__load_dataset_(X, self.batch_size) 

        # build and compile models (stage 1)
        self.__build_models(X, learning_rate)

        for index, dataset in enumerate(datasets):
                        
            # pretraining step 1:
            self.__autoencoder_pre_training(X, learning_rate, freq_warmup, index, dataset)
            
            # pretraining step 2:
            self.__synchronization_pre_trainig(sin_warmup, pos_amp, dataset)
                    
            # train anomaly detector
            with tf.device('/gpu:1'):
                for epoch in range(epochs):
                    print("\nStart of epoch %d" % (epoch,))
                    if self.synchronize:
                        for step, (x_batch, t_batch) in enumerate(dataset):
                            # Train the sine wave encoder
                            with tf.GradientTape() as tape:
                                # forward pass
                                s = self.sincoder(t_batch)
                                
                                # compute loss
                                s_loss = self.sincoder.loss(x_batch, s)  # median loss
                            
                            # retrieve gradients and update weights
                            grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                            self.sincoder.optimizer.apply_gradients(zip(grads, self.sincoder.trainable_weights))

                            # synchronize batch
                            b = self.sincoder.layers[1].wb / self.sincoder.layers[1].freq  # phase shift(s)
                            b_sync = b - tf.expand_dims(b[:,0], axis=-1)
                            th_sync = tf.expand_dims(
                                tf.expand_dims(self.sincoder.layers[1].freq, axis=0), axis=0
                            ) * (tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0))  # synchronized angle
                            e = (
                                x_batch - s
                            ) * tf.sin(
                                self.sincoder.layers[1].freq[0] * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) - b[:,0])
                            )  # noise
                            x_batch_sync = tf.reduce_sum(
                                tf.expand_dims(self.sincoder.layers[1].amp, axis=0) * tf.sin(th_sync), axis=-1
                            ) + self.sincoder.layers[1].disp + e
                            
                            # train the rancoders
                            with tf.GradientTape() as tape:
                                # forward pass
                                o_hi, o_lo = self.rancoders(x_batch_sync)
                                
                                # compute losses
                                o_hi_loss = self.rancoders.loss[0](
                                    tf.tile(tf.expand_dims(x_batch_sync, axis=0), (self.n_estimators, 1, 1)), o_hi
                                )
                                o_lo_loss = self.rancoders.loss[1](
                                    tf.tile(tf.expand_dims(x_batch_sync, axis=0), (self.n_estimators, 1, 1)), o_lo
                                )
                                o_loss = o_hi_loss + o_lo_loss

                            # retrieve gradients and update weights
                            grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                            self.rancoders.optimizer.apply_gradients(zip(grads, self.rancoders.trainable_weights))

                            upper_bound_loss = tf.reduce_mean(o_hi_loss).numpy()
                            lower_bound_loss = tf.reduce_mean(o_lo_loss).numpy()
                        print(
                            "sine_loss:", tf.reduce_mean(s_loss).numpy(), 
                            "upper_bound_loss:", upper_bound_loss, 
                            "lower_bound_loss:", lower_bound_loss, 
                            end='\r'
                        )
                    else:
                        for step, (x_batch, t_batch) in enumerate(dataset):
                        # train the rancoders
                            with tf.GradientTape() as tape:
                                # forward pass
                                o_hi, o_lo = self.rancoders(x_batch)
                                
                                # compute losses
                                o_hi_loss = self.rancoders.loss[0](
                                    tf.tile(tf.expand_dims(x_batch, axis=0), (self.n_estimators, 1, 1)), o_hi
                                )
                                o_lo_loss = self.rancoders.loss[1](
                                    tf.tile(tf.expand_dims(x_batch, axis=0), (self.n_estimators, 1, 1)), o_lo
                                )
                                o_loss = o_hi_loss + o_lo_loss

                            # retrieve gradients and update weights
                            grads = tape.gradient(o_loss, self.rancoders.trainable_weights)
                            self.rancoders.optimizer.apply_gradients(zip(grads, self.rancoders.trainable_weights))

                            upper_bound_loss = tf.reduce_mean(o_hi_loss).numpy()
                            lower_bound_loss = tf.reduce_mean(o_lo_loss).numpy()
                        print(
                            "upper_bound_loss:", upper_bound_loss, 
                            "lower_bound_loss:", lower_bound_loss, 
                            end='\r'
                        )

                    self.experiment_dataframe.loc[len(self.experiment_dataframe)] = {"actual_dataset": index,
                                                                                     "actual_epoch": epoch,
                                                                                     "sampling_rate": self.sampling_rate,
                                                                                     "learning_rate": learning_rate,
                                                                                     "batch_size": self.batch_size,
                                                                                     "n_estimators": self.n_estimators,
                                                                                     "max_features": self.max_features,
                                                                                     "encoding_depth": self.encoding_depth,
                                                                                     "latent_dim": self.latent_dim,
                                                                                     "decoding_depth": self.decoding_depth,
                                                                                     "activation": self.decoding_depth,
                                                                                     "output_activation": self.decoding_depth,
                                                                                     "delta": self.delta,
                                                                                     "synchronize": self.synchronize,
                                                                                     "upper_bound_loss": upper_bound_loss,
                                                                                     "lower_bound_loss": lower_bound_loss,
                                                                                     "AUC_ROC": None,
                                                                                    } 
    #Region: Private Methods
    def __build_models(self, X, learning_rate):
        with tf.device('/cpu:0'):
            if self.synchronize:
                self.build(X[0].shape, initial_stage = True, learning_rate = learning_rate)
                if self.freq_init:
                    self.build(X[0].shape, learning_rate = learning_rate)
            else:
                self.build(X[0].shape, learning_rate = learning_rate)

    def __synchronization_pre_trainig(self, sin_warmup, pos_amp, dataset):
        if sin_warmup > 0 and self.synchronize:
            with tf.device('/gpu:1'):
                for epoch in range(sin_warmup):
                    print("\nStart of sine representation pre-train epoch %d" % (epoch,))
                    for step, (x_batch, t_batch) in enumerate(dataset):
                            # Train the sine wave encoder
                        with tf.GradientTape() as tape:
                                # forward pass
                            s = self.sincoder(t_batch)
                                
                                # compute loss
                            s_loss = self.sincoder.loss(x_batch, s)  # median loss
                            
                            # retrieve gradients and update weights
                        grads = tape.gradient(s_loss, self.sincoder.trainable_weights)
                        self.sincoder.optimizer.apply_gradients(zip(grads, self.sincoder.trainable_weights))
                    print("sine_loss:", tf.reduce_mean(s_loss).numpy(), end='\r')
                
                # invert params (all amplitudes should either be -ve or +ve). Here we make them +ve
            if pos_amp:
                a_adj = tf.where(
                        self.sincoder.layers[1].amp[:,0] < 0, 
                        self.sincoder.layers[1].amp[:,0] * -1, 
                        self.sincoder.layers[1].amp[:,0]
                    )  # invert all -ve amplitudes
                wb_adj = tf.where(
                        self.sincoder.layers[1].amp[:,0] < 0, 
                        self.sincoder.layers[1].wb[:,0] + np.pi, 
                        self.sincoder.layers[1].wb[:,0]
                    )  # shift inverted waves by half cycle
                wb_adj = tf.where(
                        wb_adj > 2*np.pi, self.sincoder.layers[1].wb[:,0] - np.pi, wb_adj
                    )  # any cycle > freq must be reduced by half the cycle
                g_adj = tf.where(
                        self.sincoder.layers[1].amp[:,0] < 0, 
                        self.sincoder.layers[1].disp - a_adj, 
                        self.sincoder.layers[1].disp
                    )  # adjust the vertical displacements after reversing amplitude signs
                K.set_value(self.sincoder.layers[1].amp[:,0], a_adj)
                K.set_value(self.sincoder.layers[1].wb[:,0], wb_adj)
                K.set_value(self.sincoder.layers[1].disp, g_adj)

    def __autoencoder_pre_training(self, X, learning_rate, freq_warmup, index, dataset):
        if freq_warmup > 0 and self.synchronize and not self.freq_init:
            with tf.device('/gpu:1'):
                for epoch in range(freq_warmup):
                    print("\nStart of frequency pre-train epoch %d" % (epoch,))
                    for step, (x_batch, t_batch) in enumerate(dataset):
                            # Prefit the oscillation encoder
                        with tf.GradientTape() as tape:
                                # forward pass
                            z, x_pred = self.freqcoder(x_batch)
                                
                                # compute loss
                            x_loss = self.freqcoder.loss(x_batch, x_pred)  # median loss
                            
                            # retrieve gradients and update weights
                        grads = tape.gradient(x_loss, self.freqcoder.trainable_weights)
                        self.freqcoder.optimizer.apply_gradients(zip(grads, self.freqcoder.trainable_weights))
                    print("pre-reconstruction_loss:", tf.reduce_mean(x_loss).numpy(), end='\r')
                    
                # estimate dominant frequency
            with tf.device('/cpu:0'):
                z = self.freqcoder(X[index])[0].numpy().reshape(-1)  # must be done on full unshuffled series
                z = ((z - z.min()) / (z.max() - z.min())) * (1 - -1) + -1  #  scale between -1 & 1
                p = Periodogram(z, sampling=1)
                p()
                peak_idxs = find_peaks(p.psd, distance=self.min_dist, height=(0, np.inf))[0]
                peak_order = p.psd[peak_idxs].argsort()[-self.min_periods-self.max_freqs:][::-1]  # max PSDs found
                peak_idxs = peak_idxs[peak_order]
            if peak_idxs[0] < self.min_periods and not self.force_synchronization:
                self.synchronize = False
                print('no common oscillations found, switching off synchronization attempts')
            elif max(peak_idxs[:self.min_periods]) >= self.min_periods:
                idxs = peak_idxs[peak_idxs >= self.min_periods]
                peak_freqs = [p.frequencies()[idx] for idx in idxs[:min(len(idxs), self.max_freqs)]]
                self.freq_init = [2 * np.pi * f for f in peak_freqs]
                print('found common oscillations at period(s) = {}'.format([1 / f for f in peak_freqs]))
            else:
                self.synchronize = False
                print('no common oscillations found, switching off synchronization attempts')
                
                # build and compile models (stage 2)
            with tf.device('/cpu:0'):
                self.build(X[index].shape, learning_rate = learning_rate)
    
    def __normalize_data(self, X: np.ndarray, training_step: bool):
        if(training_step):
            return self.min_max_scaler.fit_transform(X)
        return self.min_max_scaler.transform(X)
        
    def __get_time_matrix(self, X):
        #each line represents an instant of time in each time series
        total_time_in_seconds, time_serie_time_instant = self.__calculate_time_serie_time_instatnt(X)
        time_matrix = np.arange(start=0, stop=total_time_in_seconds, step=time_serie_time_instant)
        return np.tile(time_matrix.reshape(-1, 1), (1, X.shape[1]))

    def __calculate_time_serie_time_instatnt(self, X):
        X_length = X.shape[0]
        total_time_in_seconds = X_length/self.sampling_rate
        time_serie_time_instant = total_time_in_seconds/self.sampling_rate
        return total_time_in_seconds,time_serie_time_instant

    def __load_dataset_(self, X: np.ndarray, batch_size: int, training_step: bool = True):
        datasets = []
        x_2D = []
        
        for x in X:
            
            if(self.mono):
                x = x.reshape(-1, 1)[:self.sampling_rate]
            else:
                x = x.T[:self.sampling_rate]
    
            x_normalized = self.__normalize_data(x, training_step)
            x_2D.append(x_normalized)
            time_matrix = self.__get_time_matrix(x)
            with tf.device('/cpu:0'):
                dataset = tf.data.Dataset.from_tensor_slices((x_normalized.astype(np.float32), time_matrix.astype(np.float32)))
                datasets.append(dataset.batch(batch_size))  
                    
        return datasets, x_2D
    
    def __get_experiment_dataframe(self):
        parameters_columns = ["actual_dataset",
                              "actual_epoch",
                              "sampling_rate",
                              "learning_rate",
                              "batch_size",
                              "n_estimators",
                              "max_features",
                              "encoding_depth",
                              "latent_dim",
                              "decoding_depth",
                              "activation",
                              "output_activation",
                              "delta",
                              "synchronize",
                              "upper_bound_loss",
                              "lower_bound_loss",
                              "AUC_ROC",
                            ]               
        return pd.DataFrame(columns=parameters_columns)
    #End region
                 
    #Region: Public methods               
    def predict(self, x: np.ndarray, desync: bool = False):
        datasets, x = self.__load_dataset_(x, self.batch_size, False)
        batches = int(np.ceil(x[0].shape[0] / self.batch_size))
        
        for index, dataset in enumerate(datasets):
            # loop through the batches of the dataset.
            with tf.device('/gpu:1'):
                if self.synchronize:
                    s, x_sync, o_hi, o_lo = [None] * batches, [None] * batches, [None] * batches, [None] * batches
                    for step, (x_batch, t_batch) in enumerate(dataset):
                        s_i = self.sincoder(t_batch).numpy()
                        b = self.sincoder.layers[1].wb / self.sincoder.layers[1].freq  # phase shift(s)
                        b_sync = b - tf.expand_dims(b[:,0], axis=-1)
                        th_sync = tf.expand_dims(
                            tf.expand_dims(self.sincoder.layers[1].freq, axis=0), axis=0
                        ) * (tf.expand_dims(t_batch, axis=-1) + tf.expand_dims(b_sync, axis=0))  # synchronized angle
                        e = (
                            x_batch - s_i
                        ) * tf.sin(
                            self.sincoder.layers[1].freq[0] * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) - b[:,0])
                        )  # noise
                        x_sync_i = (tf.reduce_sum(
                            tf.expand_dims(self.sincoder.layers[1].amp, axis=0) * tf.sin(th_sync), axis=-1
                        ) + self.sincoder.layers[1].disp + e).numpy()  
                        o_hi_i, o_lo_i = self.rancoders(x_sync_i)
                        o_hi_i, o_lo_i = tf.transpose(o_hi_i, [1,0,2]).numpy(), tf.transpose(o_lo_i, [1,0,2]).numpy()
                        if desync:
                            o_hi_i, o_lo_i = self.predict_desynchronize(x_batch, x_sync_i, o_hi_i, o_lo_i)
                        s[step], x_sync[step], o_hi[step], o_lo[step]  = s_i, x_sync_i, o_hi_i, o_lo_i
                    return (
                        np.concatenate(s, axis=0), 
                        np.concatenate(x_sync, axis=0), 
                        np.concatenate(o_hi, axis=0), 
                        np.concatenate(o_lo, axis=0)
                    )
                else:
                    o_hi, o_lo = [None] * batches, [None] * batches
                    for step, (x_batch, t_batch) in enumerate(dataset):
                        o_hi_i, o_lo_i = self.rancoders(x_batch)
                        o_hi_i, o_lo_i = tf.transpose(o_hi_i, [1,0,2]).numpy(), tf.transpose(o_lo_i, [1,0,2]).numpy()
                        o_hi[step], o_lo[step]  = o_hi_i, o_lo_i
                    return np.concatenate(o_hi, axis=0), np.concatenate(o_lo, axis=0)
    
    def predict_desynchronize(self, x: np.ndarray, x_sync: np.ndarray, o_hi: np.ndarray, o_lo: np.ndarray):
        if self.synchronize:
            E = (o_hi + o_lo)/ 2  # expected values
            deviation = tf.expand_dims(x_sync, axis=1) - E  # input (synchronzied) deviation from expected
            deviation = self.desynchronize(deviation)  # desynchronize
            E = tf.expand_dims(x, axis=1) - deviation  # expected values in desynchronized form
            offset = (o_hi - o_lo) / 2  # this is the offet from the expected value
            offset = abs(self.desynchronize(offset))  # desynch
            o_hi, o_lo = E + offset, E - offset  # add bound displacement to expected values
            return o_hi.numpy(), o_lo.numpy()  
        else:
            raise ParameterError('synchronize', 'parameter not set correctly for this method')
    
    def desynchronize(self, e: np.ndarray):
        if self.synchronize:
            b = self.sincoder.layers[1].wb / self.sincoder.layers[1].freq  # phase shift(s)
            return e * tf.sin(
                self.sincoder.layers[1].freq[0] * ((np.pi / (2 * self.sincoder.layers[1].freq[0])) + b[:,0])
            ).numpy()
        else:
            raise ParameterError('synchronize', 'parameter not set correctly for this method')
    
    def score_samples(self, X):
        sins, synched, upper, lower = self.predict(X)
        synched_tiles = np.tile(synched.reshape(synched.shape[0], 1, synched.shape[1]), (1, self.n_estimators, 1))
        result = np.where((synched_tiles < lower) | (synched_tiles > upper), 1, 0)
        return np.mean(result)
    
    def save(self, filepath: str = os.path.join(os.getcwd(), 'ransyncoders.z')):
        file = {'params': self.get_config()}
        if self.synchronize:
            file['frequencycoder'] = {'model': self.freqcoder.to_json(), 'weights': self.freqcoder.get_weights()}
            file['sinusoidalcoder'] = {'model': self.sincoder.to_json(), 'weights': self.sincoder.get_weights()}
        file['rancoders'] = {'model': self.rancoders.to_json(), 'weights': self.rancoders.get_weights()}
        dump(file, filepath, compress=True)
    
    @classmethod
    def load(cls, filepath: str = os.path.join(os.getcwd(), 'ransyncoders.z')):
        file = load(filepath)
        cls = cls()
        for param, val in file['params'].items():
            setattr(cls, param, val)
        if cls.synchronize:
            cls.freqcoder = model_from_json(file['frequencycoder']['model'], custom_objects={'FrequencyCoder': FrequencyCoder})
            cls.freqcoder.set_weights(file['frequencycoder']['weights'])
            cls.sincoder = model_from_json(file['sinusoidalcoder']['model'], custom_objects={'SinusoidalCoder': SinusoidalCoder})
            cls.sincoder.set_weights(file['sinusoidalcoder']['weights'])
        cls.rancoders = model_from_json(file['rancoders']['model'], custom_objects={'RANCoders': RANCoders})  
        cls.rancoders.set_weights(file['rancoders']['weights'])
        return cls
    
    def get_config(self):
        return self.experiment_dataframe
    #End region
        
           
# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


class RANSynCoders(BaseEstimator, OutlierMixin):
    def __init__(self, 
                 n_estimators: int = 5, 
                 max_features: int = 5, 
                 encoding_depth: int = 1, 
                 latent_dim: int = 1,  
                 decoding_depth: int = 2,
                 activation: str = 'relu', 
                 output_activation: str = 'sigmoid', 
                 delta: float = 0.05,  # quantile bound for regression
                 # Syncrhonization inputs
                 synchronize: bool = True, #False
                 force_synchronization: bool = True,  # if synchronization is true but no significant frequencies found
                 min_periods: int = 3,  # if synchronize and forced, this is the minimum bound on cycles to look for in train set
                 freq_init: Optional[List[float]] = None,  # initial guess for the dominant angular frequency
                 max_freqs: int = 5,  # the number of sinusoidal signals to fit = 1
                 min_dist: int = 60,  # maximum distance for finding local maximums in the PSD
                 trainable_freq: bool = False,  # whether to make the frequency a variable during layer weight training
                 bias: bool = True,  # add intercept (vertical displacement)
                 sampling_rate: int = 16000,
                 mono: bool = True,
                ) -> None:
            super().__init__()
            # Rancoders inputs:
            self.n_estimators = n_estimators
            self.max_features = max_features
            self.encoding_depth = encoding_depth
            self.latent_dim = latent_dim
            self.decoding_depth = decoding_depth
            self.activation = activation
            self.output_activation = output_activation
            self.delta = delta
        
            # Syncrhonization inputs
            self.synchronize = synchronize
            self.force_synchronization = force_synchronization
            self.min_periods = min_periods
            self.freq_init = freq_init  # in radians (angular frequency)
            self.max_freqs = max_freqs
            self.min_dist = min_dist
            self.trainable_freq = trainable_freq
            self.bias = bias
            self.sampling_rate = sampling_rate
            self.mono = mono
            self.model = self.build_model()

    @property
    def name(self):
        return "RANSynCoder " + "+".join([f[0] for f in self.features])
        
    def fit(self, 
            X, 
            y = None, 
            timestamps_matrix: np.ndarray = None,
            batch_size: int = 180, 
            learning_rate: float = 0.001,
            epochs: int = 10,
            freq_warmup: int = 5,  # number of warmup epochs to prefit the frequency
            sin_warmup: int = 5,  # number of warmup epochs to prefit the sinusoidal representation 
            pos_amp: bool = True,  # whether to constraint amplitudes to be +ve only
        ):
        return self.model.fit(X, 
                              y, 
                              final_model__timestamps_matrix = timestamps_matrix,
                              final_model__batch_size = batch_size, 
                              final_model__learning_rate = learning_rate,
                              final_model__epochs = epochs,
                              final_model__freq_warmup = freq_warmup,
                              final_model__sin_warmup = sin_warmup,
                              final_model__pos_amp = pos_amp,
                            )
    
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
    
    def save(self):
        return self.final_model.save()

    def get_config(self):
        return self.final_model.get_config()

    def get_final_model(self):
        return RANSynCodersBase(n_estimators = self.n_estimators, 
                                max_features = self.max_features, 
                                encoding_depth = self.encoding_depth, 
                                latent_dim = self.latent_dim, 
                                decoding_depth = self.decoding_depth, 
                                activation = self.activation,
                                output_activation = self.output_activation,
                                delta = self.delta,
                                synchronize = self.synchronize,
                                force_synchronization = self.force_synchronization,
                                min_periods = self.min_periods,  
                                freq_init = self.freq_init,  
                                max_freqs = self.max_freqs,  
                                min_dist = self.min_dist, 
                                trainable_freq = self.trainable_freq,  
                                bias = self.bias,
                                sampling_rate = self.sampling_rate,
                                mono = self.mono,
                            )
        
    def build_model(self):
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
        self.final_model = self.get_final_model()

        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("final_model", self.final_model),
                ]
            )
        
        return model