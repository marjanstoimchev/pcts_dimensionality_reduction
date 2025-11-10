# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:10:09 2022

@author: MarjanS
"""

import arff
import config
from config import *
from other_imports import *


def unselectColumn(L, L1): 
    for l in L: 
        yield [x for i, x in enumerate(l) if i not in L1] 
        
class CreateDF(ConfigSelector):
    def __init__(self, features_path, targets_path, dataset, feature_type, name, ensemble_method):
        super(CreateDF, self).__init__()
        
        self.features_path = features_path
        self.targets_path = targets_path
        self.name = name
        self.ensemble_method = ensemble_method
        self.dataset = dataset
        self.feature_type = feature_type
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
        
        
        self.out_dim = self.out_features[self.name]

        self.settings_dict = {"[General]": ["Verbose = 0"],
                              
                        "[Data]":["File = train_features.arff", 
                                    "PruneSet = train_features.arff", 
                                    "TestSet = test_features.arff",
                                    "XVal = 10"],
                                
                        "[Tree]": ["FTest = 1.0"],
                        
                        "[Model]": ["MinimalWeight = 1.0"],
                        
                        "[Output]": ["WritePredictions = Test"],
                        
                        "[Attributes]": [f"Target = {self.out_dim + 1}-{self.out_dim + 1 + self.n_classes - 1}",
                                         f"Clustering = {self.out_dim + 1}-{self.out_dim + 1 + self.n_classes - 1}",
                                         f"Descriptive = 1-{self.out_dim}"],
                        
                        "[Ensemble]": ["Iterations = 150",
                                        f"EnsembleMethod = {self.ensemble_method}",
                                        "SelectRandomSubspaces = SQRT"]
                                        #"NumberOfThreads = 16"
                                }     
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
        
        n = "full_split"        
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
        store_path = ["arffs", f"{self.ensemble_method}", f"{self.dataset}", f"{self.feature_type}", f"{self.name}", f"{n}"]
        store_path = "/".join(store_path) 
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        
        for split in splits:
            file_name = "/".join([store_path, f"{split}_features.arff"])
            with open(file_name, "w", newline='') as f:
                print(f"@relation {self.dataset}", file=f)
                for i in range(df['features'][split].shape[1]):        
                    print(self._arff_header_row(i, True, "numeric"), file=f)
                for j in range(df['targets'][split].shape[1]):
                    print(self._arff_header_row(i+1+j, False, "{0, 1}"), file=f)
                print("@data", file=f)
                self._create_arff_dense(df['features'][split], df['targets'][split], f)
        self.create_settings(store_path)
    
    def create_settings(self, path):
        with open(path + "/features.s", 'w') as f: 
            for key, value in self.settings_dict.items(): 
                print('\n%s' % key, file=f)
                for v in value:
                    print('%s' % v, file=f)

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)

