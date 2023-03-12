'''

'''
# Disable Logging
from mtsa import (
    MFCCMix,
    Hitachi,
    FEATURES,
    files_train_test_split,
    calculate_aucroc,
    get_tsne_results,
    files_train_test_split_combined
)
from sklearn.model_selection import LeaveOneOut
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from functools import reduce, partial
import pandas as pd
import itertools as ite
import numpy as np
import argparse
from common import elapsed_time, multiple_runs
import tensorflow as tf
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
tf.keras.utils.disable_interactive_logging()
import re
import json

def convert(match_obj):
            if match_obj.group(0) == '_':
                return ""
            if match_obj.group(0) == '/':
                return " "

class PaperScript2023:

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def results(self):
        # self.results_individual()
        self.results_combined()

    def get_features(self):
        number_features = np.arange(1, len(FEATURES)+1)
        def combinations(r): return ite.combinations(FEATURES, r)
        all_combinations = map(combinations, number_features)
        features = reduce(lambda x, y: ite.chain(x, y), all_combinations)
        return features

    def get_models_mfccmix(self):
        def build_mfccmix_by_features(features):
            mfccmix = MFCCMix(
                features=list(features),
                final_model=GaussianMixture(n_components=self.n_components)
            )
            return ("MFCCMix " + "+".join([f[0] for f in features]), mfccmix)
        features = self.get_features()
        mfcc_models = map(build_mfccmix_by_features, features)
        return mfcc_models

    def get_models_baseline(self):
        hitachi = [("Baseline", Hitachi())]
        return hitachi

    def get_models(self):
        model_baseline = self.get_models_baseline()
        models_mfccmix = self.get_models_mfccmix()
        all_models = reduce(lambda x, y: ite.chain(
            x, y), (model_baseline, models_mfccmix))
        return list(all_models)

    def get_paths(self, level):
        paths = glob.glob(os.path.join(self.path, f'*{os.sep}' * level))
        return paths

    def write_dict(self, my_dict, **kwargs):
        path_output = os.path.join(self.output, 'rocs')
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        filename = 'filename'
        if 'filename' in kwargs:
            filename = kwargs['filename']
        filepath = os.path.join(path_output, f'{filename}')
        with open(f'{filepath}.tex', 'w') as tf:
            json.dump(my_dict, tf)
            
    def write_df(self, df_final, **kwargs):
        path_output = os.path.join(self.output, 'rocs')
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        filename = 'filename'
        if 'filename' in kwargs:
            filename = kwargs['filename']
        filepath = os.path.join(path_output, f'{filename}')
        # with open(f'{filepath}.tex', 'w') as tf:
        #     # tf.write(df_final.to_latex())
        #     tf.write(df_final.style.to_latex())
        df_final.to_hdf(f'{filepath}.h5', 'df')

    def convert_to_dataframe(self, rocs, level):
        rocs_norms = list(map(self.extract_params, [(r, level) for r in rocs]))
        # features = list(self.get_paths(level))]
        df = pd.DataFrame(rocs_norms)
        col_params = [f'P{p}' for p in range(level+1)]
        col_features = ['features', 'roc']
        df_params = pd.DataFrame(
            df[0].tolist(), index=df.index, columns=col_params)
        df_rocs = pd.DataFrame(df[[1, 2]])
        df_rocs.columns = col_features
        df_final = pd.concat([df_params, df_rocs], axis=1)
        df_final = pd.pivot(df_final, index=list(
            df_params.columns), columns='features', values='roc')
        df_final = df_final[sorted(df_final.columns, key=len)]
        df_final = df_final.loc[df_final.index].apply(lambda x: round(x, 2))
        return df_final

    def extract_params(self, params):
        roc, level = params
        params = roc[0].split(os.sep)[:-1][-(level+1):]
        return tuple(params), roc[1], roc[2]

    def fit_individual_model(self, params):
        path, model_params = params
        _, model = model_params
        X_train, X_test, y_train, y_test = files_train_test_split(path)
        result_fit = elapsed_time(model.fit, X_train, y_train)
        result_roc = elapsed_time(calculate_aucroc, model, X_test, y_test)
        x, y, hue = self.local_get_tsne_results(model, X_test, y_test)
        # self.plot_tsne(model_name, path, x, y, hue)
        result = {
            'results_fit': result_fit,
            'results_roc': result_roc,
            'results_tsne': (x, y, hue)
        }
        return result

    def local_get_tsne_results(self, model, X_test, y_test):
        x, y, hue = get_tsne_results(
            model=model,
            X=model.transform(X_test),
            y=y_test,
            perplexity=self.perplexity
        )
        return  x, y, hue
    
    def plot_tsne(self, model_name, path, x, y, hue):
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=x, y=y,
            hue=hue,
            palette=sns.color_palette("hls", 2),
            legend="full",
            s=100
            # alpha=0.3
        )
        path_output = os.path.join(self.output, 'results_02_tse')
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        filename = '_'.join(path.split(os.path.sep)[
                            :-1][-(self.level+1):]) + "_" + model_name
        plt.savefig(os.path.join(path_output, f'{filename}.pdf'))

    def plot_tsne(self, params):
            filename, x, y, hue = params
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=x, y=y,
                hue=hue,
                palette=sns.color_palette("hls", 2),
                legend="full",
                s=100
                # alpha=0.3
            )
            path_output = os.path.join(self.output, 'results_02_tse')
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            plt.savefig(os.path.join(path_output, f'{filename}.pdf'))

    
        

    def results_individual(self):
        print("START: results_01_individual")
        
        all_models = self.get_models()
        all_paths = self.get_paths(self.level)
        all_params = list(ite.product(all_paths, all_models))
        runs = self.runs
        results = [multiple_runs(runs, self.fit_individual_model, param) for param in all_params]        
        
        
        df = pd.DataFrame.from_dict(
            {
                (re.sub(r"_|/", convert , param[0][0]).strip(), 
                 param[0][1][0],
                 run['run_id']
                 ): 
                    {
                        "fit time": run['run_details']['fun_return']['results_fit']['time_elapsed'],
                        "roc time": run['run_details']['fun_return']['results_roc']['time_elapsed'],
                        "roc value": run['run_details']['fun_return']['results_roc']['fun_return']
                    }
                for param in list(zip(all_params, results))
                for run in param[1]
             
            },
            orient='index'
        )

        df.index = df.index.set_names(['path', 'model', 'run id'])

        def get_filename(param, run):
            path = param[0]
            model_name = param[1][0]
            run_id = run['run_id']
            filename = f"{'_'.join(path.split(os.path.sep)[:-1][-(self.level+1):])}_{model_name}_{run_id}"
            return filename

        
        list(
            map(
                self.plot_tsne, 
                [
                    (get_filename(param[0], run),
                    *run['run_details']['fun_return']['results_tsne']) 
                    
                    for param in list(zip(all_params, results)) 
                    for run in param[1]
                ]
                )
            )
        
        self.write_df(df, filename='results_01_individual')
        
        print("FINISH: results_01_individual")


    def results_combined(self):
        print("START: results_03_combined")

        def calculate_roc(params):
            paths, model_name, model, split = params
            train = paths[split[0]]
            test = paths[split[1]]
            print(
                f"results_03_combined calculate_roc {str(test)} {model_name}")
            train_X_train, train_X_test, train_y_train, train_y_test = files_train_test_split_combined(
                train)
            test_X_train, test_X_test, test_y_train, test_y_test = files_train_test_split_combined(
                test)
            model.fit(train_X_train, train_y_train)
            roc = calculate_aucroc(model, test_X_test, test_y_test)
            return str(test[0]), model_name, roc

        def calculate_roc_combined(params):
            path, model_params = params
            model_name, model = model_params
            print(
                f"results_03_combined calculate_roc_combined {path} {model_name}")
            paths = np.array(glob.glob(os.path.join(path, f'*{os.sep}')))
            loo = LeaveOneOut()
            splits = loo.split(paths)
            rocs = map(calculate_roc, [
                       (paths, model_name, model, split) for split in splits])
            return list(rocs)

        all_paths = self.get_paths(self.level - 1)
        all_models = self.get_models()
        all_params = list(ite.product(all_paths, all_models))
        
        results = [
            multiple_runs(self.runs, 
            calculate_roc_combined, 
            param) 
            for param in all_params
            ]   
        

        df = pd.DataFrame.from_dict(
            {
                (re.sub(r"_|/", convert , test[0]).strip(), 
                 test[1],
                 machine['run_id']
                 ): 
                    {
                        "roc value": test[2]
                    }
                for param in list(zip(all_params, results))
                for machine in param[1]
                for test in machine['run_details']['fun_return']
             
            },
            orient='index'
        )

        df.index = df.index.set_names(['path', 'model', 'run id'])

        self.write_df(df, filename='results_03_combined')
        print("FINISH: results_03_combined")


def main(parse_args):
    paper_script_2023 = PaperScript2023(**parse_args.__dict__)
    paper_script_2023.results()


if __name__ == "__main__":
    print("MTSA Pipeline")
    print("Initializing...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', type=int)
    parser.add_argument('--path', type=str)
    parser.add_argument('--n_components', type=int)
    parser.add_argument('--level', type=int)
    parser.add_argument('--perplexity', type=int)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--output', type=str)
    parse_args = parser.parse_args()
    main(parse_args)
    print("MTSA Pipeline: Finished")
