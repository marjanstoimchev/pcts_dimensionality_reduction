# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:39:28 2022

@author: MarjanS
"""

import arff
import config
from config import *
from other_imports import *
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker

class CreateSettingsSSL(ConfigSelector):
    def __init__(self,   arffs_path,
                         features_path,
                         targets_path,
                         dataset, 
                         feature_type,
                         model_name, 
                         percentage_labeled, 
                         apply_ensemble, 
                         apply_ssl,
                         seed,
                         ):
        
        super(CreateSettingsSSL, self).__init__()
        
        self.arffs_path = arffs_path
        self.features_path = features_path
        self.targets_path = targets_path
        
        self.dataset = dataset
        self.feature_type = feature_type
        self.model_name = model_name
        self.percentage_labeled = percentage_labeled
        
        self.apply_ensemble = apply_ensemble
        self.apply_ssl = apply_ssl
        self.seed = seed
        
        self.config = self.select(self.dataset)
        self.n_classes = self.config.n_classes
        self.out_features = {  "Vgg16": 4096,
                               "Vgg19": 4096,
                               "resnet34": 512,
                               "resnet50": 2048,
                               "resnet152": 2048,
                               "effnet_b0": 1280,
                               "effnet_b1": 1280,
                               "effnet_b2": 1408,
                            }

        self.out_dim = self.out_features[self.model_name]
        
        if self.apply_ssl:
            self.settings_dict = {"[General]": [f"RandomSeed = {self.seed}",
                                                 "Verbose = 1"],
                                  
                                "[Data]":["File = ../../../train_features.arff", 
                                          "TestSet = ../../../test_features.arff"],
                                        
                                "[Tree]": ["Heuristic = VarianceReduction"],
                                
                                "[Output]": ["TrainErrors = Yes",
                                             "TestErrors = Yes",
                                             "WritePredictions = Test"],
                                
                                "[Attributes]": [f"Target = {self.out_dim + 1}-{self.out_dim + 1 + self.n_classes - 1}",
                                                 f"Descriptive = 1-{self.out_dim}"],              
                                                  
                                "[SemiSupervised]": ["SemiSupervisedMethod = PCT",
                                                     f"PercentageLabeled = {self.percentage_labeled}",
                                                     "InternalFolds = 4",
                                                     ]
                                }
            
            self.learning_setting = "ssl"
            
            
        else: 
            self.settings_dict = {"[General]": [f"RandomSeed = {self.seed}",
                                                 "Verbose = 1"],
                                  
                                "[Data]":["File = ../../../train_features.arff", 
                                          "TestSet = ../../../test_features.arff"],
                                        
                                "[Tree]": ["Heuristic = VarianceReduction"],
                                
                                "[Output]": ["TrainErrors = Yes",
                                             "TestErrors = Yes",
                                             "WritePredictions = Test"],
                                
                                "[Attributes]": [f"Target = {self.out_dim + 1}-{self.out_dim + 1 + self.n_classes - 1}",
                                                 f"Clustering = {self.out_dim + 1}-{self.out_dim + 1 + self.n_classes - 1}",
                                                 f"Descriptive = 1-{self.out_dim}"],
                                                                                      
                                "[SemiSupervised]": ["SemiSupervisedMethod = PCT",
                                                     f"PercentageLabeled = {self.percentage_labeled}",
                                                     "InternalFolds = 4",
                                                     "PossibleWeights = 1",
                                                     ]
                                }
            
            self.learning_setting = "sl"
            
        
        if self.apply_ensemble:
            self.settings_dict["[Ensemble]"] = ["Iterations = 100",
                                                f"EnsembleMethod = RForest",
                                                "SelectRandomSubspaces = SQRT",
                                                "NumberOfThreads = 8"]
            
            self.settings_dict["[SemiSupervised]"].insert(1, "PruningWhenTuning = No")
            self.method = "RForest"
        
        else:
            self.method = "PCT"
            self.settings_dict['[Data]'].insert(1, "PruneSet = ../../../train_features.arff")
            self.settings_dict["[SemiSupervised]"].insert(1, "PruningWhenTuning = Yes")
            

    # New methods #
    @staticmethod
    def _create_arff_dense(xs, ys, file_handle):
        if len(xs) != len(ys):
            raise ValueError(f"len(features) = {len(xs)} != {len(ys)} = len(targets)")
        for x, y in zip(xs, ys):
            line = ",".join(str(c) for space in [x, y] for c in space)
            print(line, file=file_handle)
            
    @staticmethod
    def _arff_header_row(i, is_feature, values):
        column_name = f"feature{i + 1}" if is_feature else f"target{i + 1}"
        return f"@attribute {column_name} {values}"

    def read_data(self, f_path, t_path):
        features = np.load(f_path)
        #features = np.round(features.astype(float), 2)
        targets = np.load(t_path).astype(int).astype(str)
        return features, targets
            
    def create(self):                
    
        splits = ['train', 'val', 'test']
        df = defaultdict(dict)
        
        for i, split in enumerate(splits):
            features, targets = self.read_data(self.features_path[i], self.targets_path[i])   
            df['features'][split] = features
            df['targets'][split] = targets
            
        df['features']['train'] = np.concatenate([df['features']['train'],
                                                  df['features']['val']])
        
            
        df['targets']['train'] = np.concatenate([df['targets']['train'],
                                                  df['targets']['val']])
        
        splits.remove('val')
        
        store_path_data =    [f"{self.arffs_path}", 
                              f"{self.method}",
                              f"{self.dataset}",
                              f"{self.feature_type}", 
                              f"{self.model_name}"]  
        
        store_path_settings = [f"{self.arffs_path}", 
                              f"{self.method}",
                              f"{self.dataset}",
                              f"{self.feature_type}", 
                              f"{self.model_name}", 
                              f"seed_{self.seed}",
                              f"labeled_{self.percentage_labeled}",
                              f"{self.learning_setting}"]
        
        
        store_path_data = "/".join(store_path_data) 
        store_path_settings = "/".join(store_path_settings) 
        
        if not os.path.exists(store_path_data):
            os.makedirs(store_path_data)
 
        if not os.path.exists(store_path_settings):
            os.makedirs(store_path_settings)
        
        self.create_settings(store_path_settings)
        
        # for split in splits:
        #     file_name = "/".join([store_path_data, f"{split}_features.arff"])     
        #     if os.path.exists(file_name):
        #         self.info_message("The path: {} exists...skipping. Creating only settings file.", file_name)
        #         continue
                
        #     else:
        #         with open(file_name, "w", newline='') as f:
        #             print(f"@relation {self.dataset}", file=f)
        #             for i in range(df['features'][split].shape[1]):        
        #                 print(self._arff_header_row(i, True, "numeric"), file=f)
        #             for j in range(df['targets'][split].shape[1]):
        #                 print(self._arff_header_row(i+1+j, False, "{0, 1}"), file=f)
        #             print("@data", file=f)
        #             self._create_arff_dense(df['features'][split], df['targets'][split], f)
        
        
    def create_settings(self, path):
        with open(path + "/features.s", 'w') as f: 
            for key, value in self.settings_dict.items(): 
                print('\n%s' % key, file=f)
                for v in value:
                    print('%s' % v, file=f)
        
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


class PlotCurves(object):
    def __init__(self, root_dir, methods, metrics, name):
        self.root_dir = root_dir
        self.methods = methods
        self.metrics = metrics
        self.name = name
        self.metric_name = self.metrics[self.name] 
        
    def extract_metrics(self, path, method):
        values = []
        oth_names = []
        
        if method == "RForest":
            names = ['Default', 'Original']
            subsets = ["Train", "Test"]
        elif method == "PCT":
            names = ['Default', 'Original', "Pruned"]
            subsets = ["Train", "Val", "Test"]
        
        with open(path,'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith(self.metric_name):
                    for k in range(1, len(names) + 1):
                        value = lines[i+k].strip().split()
                        print(value)
                        name, r_loss = value[0], float(value[2].split('(')[0])
                        #float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", value[2])[0])
                        values.append(r_loss)

        values = np.array(values)
        values = values.reshape(len(names),len(subsets))

        df = pd.DataFrame(values)
        #print(df)
        df = df.rename(columns = {i:name for i, name in enumerate(subsets)})
        df = df.rename(index = {i:name for i, name in enumerate(names)})
        return df

    def create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path) 
        return path


    def plot(self, dataset, model_name, seeds, percents, ax, fig, font_size, save = True):

        for i, values in enumerate(self.methods.items()):
            name, value = values
            learning, method = name.split(" ")
            
            color = value[0]
            marker = value[1]
            ls = value[2]
            
            if method == "RForest":
                tree_type = 'Original'
            elif method == "PCT":
                tree_type = 'Pruned'
                
            means = []
            stds = []
            for percent in percents:
                rloss = []
                for seed in seeds:
                    path = f"{self.root_dir}/{method}/{dataset}/FineTune/{model_name}/seed_{seed}/labeled_{percent}/{learning}/features.out"
                    
                    rloss += [self.extract_metrics(path, method)['Test'][tree_type]]
                    
                rloss = np.array(rloss)
                mean, std = rloss.mean(), rloss.std()
                means.append(mean)
                stds.append(std)
                     
            means = np.array(means)
            
            
            stds = np.array(stds)
            x = range(len(means))
            
            ax.plot(means, c = color, ls = ls, lw = 0.5)
            lower = means - stds
            upper = means + stds
            #ax.fill_between(range(len(means)), lower, upper, color=color, alpha=.1)
            
            learning = "" if learning == "sl" else 'SSL-' 
            label = learning
            ax.errorbar(x, 
                        means, 
                        yerr = stds,
                        color = color, 
                        fmt='--'+marker, 
                        markerfacecolor = 'none', 
                        markersize = 10, 
                        label = learning + method)
            
            ax.set_xticks(range(len(means)))
            ax.axes.set_xticklabels(percents, rotation=0)
            plt.show()    
        
        lines_labels = ax.get_legend_handles_labels()
        lines, labels = lines_labels
        
        ph = [plt.plot([],marker="", ls="")[0]]*2
        lines = ph[:1] + lines[:4] + ph[1:] + lines[4:]
        labels = ["Method"] + labels[:4]
        
        leg = ax.legend(lines, labels,
                         loc='lower center',
                         frameon=False,
                         handlelength=1,
                         borderaxespad=1, 
                         bbox_to_anchor=(1.1, 0.7),
                         fontsize = font_size)
        
        
        
        minimum = 0 if min(means) < 0 else min(means)
        maximum = max(means)
        step = (maximum - minimum) / 2
        
        tick_marks_y = np.arange(minimum, maximum, step)
        ax.axes.set_yticks(tick_marks_y)
        
        #ax.set_ylim([minimum, maximum])
        
        print(step)
        loc = ticker.MultipleLocator(base=step) # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        ax.tick_params(axis='x', labelrotation = 0)
        ax.grid(axis='y', linestyle = "dotted", dashes=(3, 20), alpha=1)
        ax.set_xlabel('Number of labeled examples $(\%)$', fontsize=font_size+5)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_title(f"{model_name} model", fontsize = font_size+5)
        ax.set_ylabel(f"{self.name}", fontsize = font_size+5)
        
        if save:
            path_name = f'{dataset}_ssl'
            self.create_path(path_name)
            fig.savefig(f'{path_name}/{self.name}.pdf', dpi=1000, format='pdf', bbox_extra_artists=(leg,), bbox_inches='tight')


def main():

    model_types = [
        # "Vgg16",
        # "Vgg19", 
        # "resnet34", "resnet50", "resnet152", 
        # "effnet_b0", "effnet_b1", 
        "effnet_b2"
                   ]
    
    seeds = ["0", "1", "42"]
    datasets = ["AID"]
    ensemble_methods = ["RForest"]
    feature_type = "FineTune"
    percentage_labeled = ["1", "5", "10", "25"]    
    root_dir = "RESULTS/FEATURES"
    splits = ["train", "val", "test"]
    
    for em in ensemble_methods:
        for apply_ensemble in [True, False]:
            for dataset in datasets:
                for seed in seeds:
                    for model_type in model_types:   
                        for pl in percentage_labeled:
                            for apply_ssl in [True, False]:
                        
                                features_path = f"{feature_type}/{dataset}_features_{feature_type}/{model_type}"
                                features_paths = []
                                targets_paths = []
                                
                                for split in splits:
                                    features_paths += ["/".join([root_dir, features_path, f"{feature_type}_{split}_features_{model_type}.npy"])]
                                    targets_paths += ["/".join([root_dir, features_path, f"{split}_targets.npy"])]
                                
                                cs =  CreateSettingsSSL("Arfs",
                                                        features_paths,
                                                        targets_paths,
                                                        dataset, 
                                                        feature_type,
                                                        model_type, 
                                                        pl, 
                                                        apply_ensemble,
                                                        apply_ssl,
                                                        seed) 
                                                         
                                cs.info_message("Dataset: {}, seed: {}, model: {}", dataset, seed, model_type)
                                cs.create()

if __name__ == "__main__":
    main()     
#%%     

# font_size = 7
# seeds = [0, 1, 42]
# percents = [1, 5, 10, 25]
# dataset = "UCM"

# root_dir = "C:/Users/MarjanS/Desktop/Arfs/"

# metric_names = {"Ranking Loss": "RankingLoss",
#                 "One Error": "OneError", 
#                 "Coverage": "Coverage", 
#                 "Average Precision": "AveragePrecision", 
#                 "ML Accuracy": "MLAccuracy",
#                 "ML Precision": "MLPrecision", 
#                 "ML Recall": "MLRecall", 
#                 "ML F1": "MLFOneMeasure",
#                 "Subset Accuracy": "SubsetAccuracy",
#                 "Micro Precision": "MicroPrecision",
#                 "Micro Recall": "MicroRecall",
#                 "Micro F1": "MicroFOne",
#                 "average AUROC": "averageAUROC",
#                 "average AUPRC": "averageAUPRC",
#                 "weighted AUPRC": "weightedAUPRC",
#                 "pooled AUPRC": "pooledAUPRC"
#                 }

# models_names = ["Vgg16", "Vgg19", "resnet34", "resnet50", "resnet152", "effnet_b0", "effnet_b1", "effnet_b2"]
# #%%
# for metric, _ in tqdm(metric_names.items()):
    
#     fig, ax = plt.subplots(figsize = (15.15,7), nrows = 2, ncols=4, sharey=True,  constrained_layout=False)
#     fig.subplots_adjust(wspace=0.5, bottom=0.2, hspace=0.8)
#     ax = ax.ravel()
    
#     for i, model_name in enumerate(models_names):    
#         methods = {"sl RForest": ["k", 'x', 'solid'], 
#                     "ssl RForest": ["#00C957", 'o', 'solid'], 
#                     "sl PCT": ["r", 'd', '-'], 
#                     "ssl PCT": ["b", 'p', '--'], 
#                     }
        
#         pc = PlotCurves(root_dir, methods, metric_names, metric)
#         pc.plot(dataset, model_name, seeds, percents, ax[i], fig, font_size, save = True)
    
# fig.suptitle(f'{dataset} dataset')
# #%%
# import math
# N = [1, 5, 10, 25]
# t = 0
# not_em = 0
# mode = "sl"
# sizes = np.empty(shape = (len(models_names), len(seeds), len(N)))
# for i, model_name in enumerate(models_names):
#     for j, seed in enumerate(seeds):
#         for k, n_labeled in enumerate(N):
#             file = f"C:/Users/MarjanS/Desktop/Arfs/PCT/MLRSNet/FineTune/{model_name}/seed_{seed}/labeled_{n_labeled}/{mode}/features.out"
#             if os.path.exists(file):
#                 t+=1
#                 size = os.path.getsize(file) == 0
#                 #print(f"Model_name: {model_name},  seed: {seed}, N: {n_labeled} - > is empty ? {size}")
#                 not_em += size
#                 sizes[i, j, k] = size
                
# total = math.prod(sizes.shape)
# print(f"\nGenerated for mode - {mode}: {t - not_em}/{total}")
# #%%
# # pretrained = True
# # model_name = "vgg16"
# # backbone =  torch.hub.load('pytorch/vision:v0.8.2', model_name, pretrained=pretrained)
# #%%

# list_array = [3, 1, 3, 5, 10, 6, 4, 3, 1]
# N = list_array[0]
# list_array.pop(0)
# median_array = []

# i = 0
# while i < N:
#     median_array.append(i + 1)
#     i = i + 1
    
# for k in range(len(list_array) - N + 1):
#     median_array.append(np.median(np.array(sorted(list_array[k:k+N]))))

    
    
    