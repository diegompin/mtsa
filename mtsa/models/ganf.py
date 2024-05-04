from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
import random
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from functools import reduce
from sklearn.base import BaseEstimator, OutlierMixin
from functools import reduce
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from mtsa.features.mel import Array2Mfcc
from mtsa.models.GANF_components.NF import MAF, RealNVP
from mtsa.models.GANF_components.audiodata import AudioData
from mtsa.models.GANF_components.gnn import GNN
from mtsa.utils import Wav2Array
from sklearn.pipeline import Pipeline
from mtsa.utils import Wav2Array

class GANFBaseModel(nn.Module):
    def __init__ (self, 
                  n_blocks = 6,
                  input_size = 1,
                  hidden_size = 32,
                  n_hidden = 1,
                  dropout = 0.0,
                  model="MAF",
                  batch_norm=False,
                  rho = 1.0,
                  rho_max = float(1e16),
                  max_iteraction = 2,
                  learning_rate = float(1e-3),
                  alpha = 0.0,
                  weight_decay = float(5e-4),
                  epochs = 20,
                  device = None,
                  ):
        super().__init__()
        self.min = 0
        self.max = 0 
        self.adjacent_matrix = None
        self.rho = rho         
        self.rho_max = rho_max     
        self.max_iteraction = max_iteraction    
        self.learning_rate = learning_rate          
        self.alpha = alpha       
        self.weight_decay= weight_decay
        self.epochs = epochs
        self.hidden_state = None

        if device != None: 
            self.device = device
        else:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        #Reducing dimensionality 
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout, device=self.device)

        # Graph Neural Networks model
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size) 

        #probability density estimation
        if model=="MAF":
            #Masked auto regressive flow
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        else:
            #real-valued non-volume preserving (real NVP)
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm)

        self.to(device=self.device)

    @property
    def name(self):
        return "GANFBaseModel " + "+".join([f[0] for f in self.features])
      
    def get_adjacent_matrix(self):
        return self.adjacent_matrix

    def fit(self, X, y=None, batch_size = None, epochs= None, max_iteraction= None, learning_rate = None, debug_dataframe= None, isWaveData = False, mono = None):
        torch.cuda.manual_seed(10)
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        epoch_nan = None #only for script experiment
        epoch = 0
        h = None
        torch.autograd.set_detect_anomaly(True)
        self.isMonoData = mono

        loss_best = 100
        self.isWaveData = isWaveData
        if epochs is not None:
           self.epochs = epochs
        if max_iteraction is not None:
            self.max_iteraction =  max_iteraction
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if batch_size is None:
            batch_size = 32

        h_A_old = np.inf
        h_tol = float(1e-6)

        dimension = X.data.shape[1]
        self.adjacent_matrix = self.__init_adjacent_matrix(X.data)

        dataloaders = self.__create_dataLoader(torch.tensor(X), batch_size=batch_size, isMonoData = mono)
        adjacent_matrix = self.adjacent_matrix

        self.__fitCore(loss_best, h_A_old, h_tol, dimension, dataloaders, adjacent_matrix)

    def __fitCore(self, loss_best, h_A_old, h_tol, dimension, dataloaders, adjacent_matrix):
        for j in range(self.max_iteraction):
            print('iteraction ' + str(j+1) + ' of ' + str(self.max_iteraction))

            while self.rho < self.rho_max:
                learning_rate = self.learning_rate #np.math.pow(0.1, epoch // 100)    
                optimizer = torch.optim.Adam([
                    {'params': self.parameters(), 'weight_decay':self.weight_decay},
                    {'params': [adjacent_matrix]}], lr=learning_rate, weight_decay=0.0)

                for epoch in range(self.epochs):
                    loss_train = []
                    epoch += 1
                    self.train()

                    for dataloader in dataloaders:
                        for x in dataloader:
                            x = x.to(self.device)
                            
                            optimizer.zero_grad()
                            loss = -self.forward(x, adjacent_matrix)
                            h = torch.trace(torch.matrix_exp(adjacent_matrix*adjacent_matrix)) - dimension
                            total_loss = loss + 0.5 * self.rho * h * h + self.alpha * h

                            h.to(self.device)
                            
                            total_loss.backward()
                            clip_grad_value_(self.parameters(), 1)
                            optimizer.step()
                            loss_train.append(loss.item())
                            adjacent_matrix.data.copy_(torch.clamp(adjacent_matrix.data, min=0, max=1))
                            
                            if np.mean(loss_train) < loss_best:
                                loss_best = np.mean(loss_train)
                                self.adjacent_matrix = adjacent_matrix

                    print('Epoch: {}, train -log_prob: {:.2f}, h: {}'.format(epoch, np.mean(loss_train), h.item()))
                    
                del optimizer
                torch.cuda.empty_cache()

                if h.item() > 0.5 * h_A_old:
                    self.rho *= 10
                else:
                    break

            h_A_old = h.item()
            self.alpha += self.rho*h.item()

            if h_A_old <= h_tol or self.rho >= self.rho_max:
                break

    def add2DebugDataframe(self, batch_size, debug_dataframe, epoch_nan, loss_best, adjacent_matrix, epoch, loss_train, A_hat, A_hat_NaN, h, total_loss, loss_train_mean, adjacent_matrix_NaN):
        debug_dataframe.loc[len(debug_dataframe)] = {'batch_size' : batch_size, 
                                                                         'epoch_size' : self.epochs, 
                                                                         'current_epoch' : epoch, 
                                                                         'epoch_NaN' : epoch_nan, 
                                                                         'learning_rate' : [self.learning_rate],
                                                                         'A_hat' : A_hat.cpu().detach().numpy(), 
                                                                         'Adjacent_Matrix' : adjacent_matrix.cpu().detach().numpy(),
                                                                         'h' : h.cpu().detach().numpy(),
                                                                         'loss_train' : loss_train, 
                                                                         'loss_train_mean' : loss_train_mean,
                                                                         'loss_best': loss_best,
                                                                         'alpha':  self.alpha, 
                                                                         'rho': self.rho,
                                                                         'total_loss': total_loss.cpu().detach().numpy(),
                                                                         'A_hat_NaN': A_hat_NaN.cpu().detach().numpy(),
                                                                         'Adjacent_Matrix_NaN': adjacent_matrix_NaN.cpu().detach().numpy()}
    
    def predict(self, X):
        return self.forward(X, self.adjacent_matrix)

    def score_samples(self, X):
        if self.isWaveData:
            return self.__score_wave(X=X)
        return self.__score_data(X=X)

    def __score_data(self, X):
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], 1, 1)
        X = torch.tensor(X).float()
        X = X.to(self.device)
        result = self.predict(X=X)
        return float(result)
    
    def __score_wave(self, X):
        dataloaders = self.__create_dataLoader(X, window_size=1, isMonoData = self.isMonoData)
        result = []
        for dataloader in dataloaders:
            for x in dataloader:
                x = x.to(self.device)
                result.append(self.predict(X=x))
        return torch.tensor(result).mean()
       
    def __create_dataLoader(self, X, batch_size = 32, window_size = 12, isMonoData = True):
        X = self.__create_dataframe(X, isMonoData)
        dataloaders = []
        for x in X:
            x_dataloader = DataLoader(AudioData(x, window_size=window_size), batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
            dataloaders.append(x_dataloader)
        return dataloaders

    def forward(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)
        log_prob_x = log_prob.mean()
        return log_prob_x
    
    def __create_dataframe(self, X, isMonoData):
        if self.isWaveData and not isMonoData:
            dataframes = []
            for x in X:
                x = x.T
                x_df = pd.DataFrame(x)
                x_df= x_df.reset_index()
                x_df = x_df.rename(columns={"index":"channel"})
                x_df["channel"] = pd.to_datetime(x_df["channel"], unit="s")
                x_df = x_df.set_index("channel")
                x_df = self.__normalize(x_df) 
                x_df = x_df.sort_index()
                dataframes.append(x_df)
            return dataframes
        
        if self.isWaveData and isMonoData:
            X = X.T
            X_df = pd.DataFrame(X)
            X_df= X_df.reset_index()
            X_df = X_df.rename(columns={"index":"channel"})
            X_df["channel"] = pd.to_datetime(X_df["channel"], unit="s")
            X_df = X_df.set_index("channel")
            X_df = self.__normalize(X_df) 
            X_df = X_df.sort_index()
            return [X_df]
        
        X_df = pd.DataFrame(X)
        X_df= X_df.reset_index()
        X_df = X_df.rename(columns={"index":"channel"})
        X_df["channel"] = pd.to_datetime(X_df["channel"], unit="s")
        X_df = X_df.set_index("channel")
        X_df = self.__normalize(X_df) 
        X_df = X_df.sort_index()

        return [X_df]

    def __normalize(self, X):
        mean = X.values.mean()
        std = X.values.std()
        X = (X - mean)/std
        return X

    def __init_adjacent_matrix(self, X):
        torch.cuda.manual_seed(10)
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(10)
        init = torch.zeros([X.shape[1], X.shape[1]])
        init = xavier_uniform_(init).abs()
        init = init.fill_diagonal_(0.0)
        return torch.tensor(init, requires_grad=True, device=self.device)

FINAL_MODEL = GANFBaseModel()

class GANF(nn.Module, BaseEstimator, OutlierMixin):

    def __init__(self, 
                 final_model=FINAL_MODEL, 
                 sampling_rate=None,
                 random_state = None,
                 isForWaveData = False,
                 batch_size = None,
                 learning_rate = None,
                 mono = True,
                 use_array2mfcc = False
                 ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        self.isForWaveData =isForWaveData
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mono = mono
        self.final_model = FINAL_MODEL
        self.use_array2mfcc = use_array2mfcc
        self.model = self._build_model()

    @property
    def name(self):
        return "GANF " + "+".join([f[0] for f in self.features])
        
    def fit(self, X, y=None, batch_size = None, epochs= None, max_iteraction= None, learning_rate = None, debug_dataframe = None):
        mono = (self.mono and not self.use_array2mfcc)
        if batch_size is None:
            batch_size = self.batch_size
        if learning_rate is None:
            learning_rate = self.learning_rate

        return self.model.fit(X, y, 
                              final_model__batch_size=batch_size,
                              final_model__epochs=epochs,
                              final_model__max_iteraction=max_iteraction,
                              final_model__learning_rate = learning_rate,
                              final_model__debug_dataframe = debug_dataframe,
                              final_model__isWaveData = self.isForWaveData,
                              final_model__mono = mono)

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
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
    def score(self, X, Y):
        X = self.score_samples
        fpr, tpr, thresholds = metrics.roc_curve(Y, X)
        auc = metrics.auc(fpr, tpr)
        return auc
    
    def get_adjacent_matrix(self):
        return self.final_model.get_adjacent_matrix()

    def _build_model(self):
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        wav2array = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
        #mono = (self.mono and not self.use_array2mfcc)
        if self.use_array2mfcc and self.isForWaveData:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("array2mfcc", array2mfcc),
                    ("final_model", self.final_model),
                    ]
                )
        elif  self.isForWaveData:
            model = Pipeline(
                steps=[
                    ("wav2array", wav2array),
                    ("final_model", self.final_model),
                    ]
                )
        else:
            model = Pipeline(
                steps=[
                    ("final_model", self.final_model),
                    ]
                )   
            
        return model