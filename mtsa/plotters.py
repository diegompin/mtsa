from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import itertools as it
import os
import abc
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.manifold import TSNE

def _get_par(name, default, **kwargs):
    if name in kwargs: return kwargs[name]
    return default

def get_tsne_results(model, X, y, **kwargs):
    n_components = _get_par('n_components', 2, **kwargs)
    perplexity = _get_par('perplexity', 4, **kwargs)
    n_iter = _get_par('n_iter', 300, **kwargs)
    random_state = _get_par('random_state', None, **kwargs)
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    
    # Xt = model['features'].fit_transform(
    #     model['array2mfcc'].fit_transform(
    #     model['wav2array'].fit_transform(X)))
    
    tsne_results = tsne.fit_transform(X)
    p1 = tsne_results[:,0]
    p2 = tsne_results[:,1]
    p3 = y
    return p1, p2, p3
        

# def plot_tsne_individual(self, filepath, **kwargs):
#     X_train, X_test, y_train, y_test = files_train_test_split(filepath)
#     x, y, hue = self.plot_tsne(X_train, X_test, y_train, y_test, **kwargs)
#     return x, y, hue
    
    
    
class Plotter(abc.ABC):
    markers = ['D', 'o', 'x','s', '^', 'd', 'h', '+', '*', ',', '.', '1', 'p', '3', '2', '4', 'H', 'v', '8',
               '<', '>']
    # colors = ['g', 'y', 'r', 'b', 'k', 'm', 'c']\

    colors = ['#1f77b4', '#ff7f0e',  '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(kwargs)
    
    def plot(self, **kwargs):
        if "path_output" in kwargs:
            path_output = kwargs["path_output"]
            dir_output = os.path.dirname(path_output)
            if not os.path.exists(dir_output):
                os.makedirs(dir_output)
        self.plot_elem(**kwargs)
    
    @abc.abstractmethod
    def plot_elem(self, **kwargs):
        """Implement """


   

   

    @staticmethod
    def plot_correlation(correlation_network, **kwargs):
        ticks = range(1, 21)
        plotter = PlotterHeatMap(X=correlation_network)
        plotter.plot(
            cmap="RdBu",
            xticks = np.arange(len(ticks)),
            yticks = np.arange(len(ticks)),
            xticklabels = ticks,
            yticklabels = ticks,
            xlabel = "MFCC Number",
            ylabel = "MFCC Number",
            # title = kwargs["title"],
            # path_output = kwargs["path_output"]
            **kwargs
        )

        
        # fig, ax = plt.subplots()
        # im = ax.imshow(correlation_network, cmap="RdBu")

        # ax.set_xticks(np.arange(len(x_ticks)))
        # ax.set_yticks(np.arange(len(x_ticks)))

        # ax.set_xticklabels(x_ticks)
        # ax.set_yticklabels(x_ticks)

        # ax.set_xlabel("MFCC Number")
        # ax.set_ylabel("MFCC Number")


        # if "title" in kwargs:
        #     ax.set_title(kwargs["title"])
        
        # cbar=plt.colorbar(im)
        # fig.tight_layout()

        # if "path_output" in kwargs:
        #     plt.savefig(kwargs["path_output"])
        # else: 
        #     plt.show()
        # plt.close()
        

    @staticmethod
    def plot_confidence_interval(confidence_intervals, **kwargs):
        fig, ax = plt.subplots(1,1)
        colors = cycle(['r', 'g', 'b', 'y'])
        for i, ci in enumerate(confidence_intervals):
            lower = ci[0]
            upper = ci[1]
            mean = (upper + lower) / 2
            color = next(colors)
            ax.plot(i + 1, mean, marker='o', markersize=10, color=color)
            ax.vlines(x=i + 1, ymin=lower, ymax=upper, linewidth=2, color=color)
            ax.hlines(y=lower, xmin=i + 1 - 0.1, xmax=i + 1 + 0.1, linewidth=2, color=color)
            ax.hlines(y=upper, xmin=i + 1 - 0.1, xmax=i + 1 + 0.1, linewidth=2, color=color)
        
        ax.set_xticks(np.arange(1, len(confidence_intervals) + 1))
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        ax.set_ylim(-1,1)

        if "title" in kwargs:
            plt.title(kwargs["title"], size=15)

        if "xticklabels" in kwargs:
            ax.set_xticklabels(kwargs["xticklabels"])

        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"], size=20)

        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"], size=20)

        
        plt.tight_layout()
        if "path_output" in kwargs:
            plt.savefig(kwargs["path_output"])
        else: 
            plt.show()
        plt.close()

    # @abstractmethod
    # def plot_confidence_interval_matrix(confidence_intervals, **kwargs):
    #     fig, ax = plt.subplots(5,5)
    #     ijs = (ij for ij in it.product(range(20), range(20)) if ij[0] != ij[1])
    #     for (i, j) in ijs:
    #         colors = cycle(['r', 'g', 'b', 'y'])
    #         for k in range(len(confidence_intervals)):
    #             lower = confidence_intervals[k,0,i,j]
    #             upper = confidence_intervals[k,1,i,j]
    #             mean = (upper + lower) / 2
    #             color = next(colors)
    #             ax[i,j].plot(k + 1, mean, marker='o', color=color)
    #             ax[i,j].vlines(x=k + 1, ymin=lower, ymax=upper, color=color)
    #             ax[i,j].hlines(y=lower, xmin=k + 1 - 0.1, xmax=k + 1 + 0.1, linewidth=2, color=color)
    #             ax[i,j].hlines(y=upper, xmin=k + 1 - 0.1, xmax=k + 1 + 0.1, linewidth=2, color=color)
            
    #         ax[i,j].set_xticks(np.arange(1, len(confidence_intervals) + 1))
    #         ax[i,j].tick_params(axis='both', which='major')
    #         ax[i,j].set_ylim(-1,1)

    #     # if "title" in kwargs:
    #     #     plt.title(kwargs["title"], size=20)

    #     # if "labels" in kwargs:
    #     #     ax.set_xticklabels(kwargs["labels"])

    #     # ax.set_xlim(0,3)
        
    #     # ax.set_xlabel("Estimators", size=20)
    #     # ax.set_ylabel("X", size=20)
        
        
    #     if "path_output" in kwargs:
    #         plt.savefig(kwargs["path_output"])
    #     else: 
    #         plt.show()
    #     plt.close()

class PlotterActivation(Plotter):

    def plot_elem(self, **kwargs):
        ticks = range(1, 21)
        plotter_heatmap = PlotterHeatMap(X=self.X)
        plotter_heatmap.plot(
            cmap="Blues",
            xticks = np.arange(len(ticks)),
            yticks = np.arange(len(ticks)),
            xticklabels = ticks,
            yticklabels = ticks,
            xlabel = "MFCC Number",
            ylabel = "MFCC Number",
            # title = kwargs["title"],
            path_output = self.path_output,
            **kwargs
        )

class PlotterHeatMap(Plotter):

    def plot_elem(self, **kwargs):
        fig, ax = plt.subplots()

        cmap="RdBu"
        if "cmap" in kwargs:
            cmap = kwargs["cmap"]

        im = ax.imshow(self.X, cmap=cmap)

        if "xticks" in kwargs:
            ax.set_xticks(kwargs["xticks"])
        
        if "yticks" in kwargs:
            ax.set_yticks(kwargs["yticks"])

        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])

        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])

        if "xticklabels" in kwargs:
            ax.set_xticklabels(kwargs["xticklabels"])
        
        if "yticklabels" in kwargs:
            ax.set_yticklabels(kwargs["yticklabels"])

        if "title" in kwargs:
            ax.set_title(kwargs["title"])
        
        cbar=plt.colorbar(im)
        fig.tight_layout()

        if "path_output" in kwargs:
            plt.savefig(kwargs["path_output"])
        else: 
            plt.show()
        plt.close()

class PlotterROC(Plotter):

    def plot_elem(self, **kwargs):
        fpr, tpr, thresholds = metrics.roc_curve(self.y_true, self.y_pred)
        auc = metrics.auc(fpr, tpr)
        # self.plot_roc(fpr, tpr, auc, title=self.title, path_output=self.path_ouput_file)

    # def plot_roc(self, **kwargs):
        
        title = ""
        if "title" in kwargs:
            title = kwargs["title"]

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='deeppink', linewidth=4, label=f"ROC Curve (area = {round(auc,2) })")
        ax.plot([0,1], [0,1], "k--",linewidth=4, label='random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
        ax.set_title(f"Receiver Operating Characteristic \n {title}", fontsize=18)
        ax.legend(loc="lower right")
        if "path_output" in kwargs:
            plt.savefig(kwargs["path_output"])
        plt.close()
