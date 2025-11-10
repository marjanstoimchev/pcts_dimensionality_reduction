# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:28:20 2022

@author: MarjanS
"""

import arff
from utils import *
from configs import *
from other_imports import *

class CreateSettingsSSL(ConfigSelector):
    def __init__(self,   arffs_path,
                         features_path,
                         targets_path,
                         dataset, 
                         model_name, 
                         percentage_labeled, 
                         method, 
                         learning_setting,
                         learning_task,
                         apply_pca,
                         seed,
                         ):
        
        super(CreateSettingsSSL, self).__init__()
        
        self.arffs_path = arffs_path
        self.features_path = features_path
        self.targets_path = targets_path
        
        self.dataset = dataset
        self.model_name = model_name
        self.percentage_labeled = percentage_labeled
        self.method = method
        self.learning_setting = learning_setting
        self.learning_task = learning_task
        self.apply_pca = apply_pca
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
        settings_dict = self.create_general_settings()
        settings_dict = self.learning_task_setting(settings_dict)
        self.settings_dict = self.create_settings(settings_dict)
        
    def create_general_settings(self):
    
        settings_dict = {"[General]": [f"RandomSeed = {self.seed}",
                                        "Verbose = 1"],
                                
                            "[Data]":["File = ../../../train_features.arff", 
                                        "TestSet = ../../../test_features.arff"],
                                    
                            "[Tree]": ["Heuristic = VarianceReduction"],
                            
                            "[Output]": ["TrainErrors = Yes",
                                            "TestErrors = Yes",
                                            "WritePredictions = Test"],
                            
                            "[Attributes]": [f"Descriptive = 1-{self.out_dim}"],     

                            "[SemiSupervised]": ["SemiSupervisedMethod = PCT",
                                                f"PercentageLabeled = {self.percentage_labeled}",
                                                "InternalFolds = 4",
                                                ]         
                                                
                            }

        
        return settings_dict
    
    def learning_task_setting(self, settings_dict):

        if self.learning_task == "mlc":
            settings_dict["[Attributes]"].insert(0, f"Target = {self.out_dim + 1}-{self.out_dim + 1 + self.n_classes - 1}")
 
        elif self.learning_task == "mcc":
            settings_dict["[Attributes]"].insert(0, f"Target = {self.out_dim + 1}")
            
        return settings_dict
    
    def create_settings_ssl_pct(self, settings_dict):
        ''' 
        PCT + SSL
        '''
        settings_dict['[Data]'].insert(1, "PruneSet = ../../../train_features.arff")
        settings_dict["[SemiSupervised]"].insert(1, "PruningWhenTuning = Yes")
        return settings_dict
    
    def create_settings_ssl_rf(self, settings_dict):
        ''' 
        RF + SSL
        '''
        settings_dict["[Ensemble]"] = ["Iterations = 100",
                                        f"EnsembleMethod = RForest",
                                        "SelectRandomSubspaces = SQRT",
                                        "NumberOfThreads = 8"]         
        settings_dict["[SemiSupervised]"].insert(1, "PruningWhenTuning = No")
        return settings_dict

    def create_settings_sl_pct(self, settings_dict):
        ''' 
        PCT + SL
        '''
        
        if self.learning_task == "mcc":
            n_clus = self.out_dim + 2
        elif self.learning_task == "mlc":
            n_clus = self.out_dim + 1 + self.n_classes - 1
            
        #settings_dict["[Attributes]"].insert(1, f"Clustering = {self.out_dim + 1}-{n_clus}")
        settings_dict['[Data]'].insert(1, "PruneSet = ../../../train_features.arff")
        settings_dict["[SemiSupervised]"].insert(1, "PruningWhenTuning = Yes")
        settings_dict["[SemiSupervised]"].insert(2, "PossibleWeights = 1")
        
        return settings_dict
    
    def create_settings_sl_rf(self, settings_dict):
        ''' 
        RF + SL
        '''

        if self.learning_task == "mcc":
            n_clus = self.out_dim + 2
        elif self.learning_task == "mlc":
            n_clus = self.out_dim + 1 + self.n_classes - 1
            
        settings_dict["[Ensemble]"] = ["Iterations = 100",
                                        f"EnsembleMethod = RForest",
                                        "SelectRandomSubspaces = SQRT",
                                        "NumberOfThreads = 8"]
        settings_dict["[SemiSupervised]"].insert(1, "PruningWhenTuning = No")
        settings_dict["[SemiSupervised]"].insert(2, "PossibleWeights = 1")

        #settings_dict["[Attributes]"].insert(1, f"Clustering = {self.out_dim + 1}-{n_clus}")
        return settings_dict

    def create_settings(self, settings_dict):
        
        if self.method == "PCT" and self.learning_setting == "ssl": 
            self.create_settings_ssl_pct(settings_dict)
        elif self.method == "RForest" and self.learning_setting == "ssl": 
            self.create_settings_ssl_rf(settings_dict)
        elif self.method == "PCT" and self.learning_setting == "sl":  
            self.create_settings_sl_pct(settings_dict)
        elif self.method == "RForest" and self.learning_setting == "sl":  
            self.create_settings_sl_rf(settings_dict)
        return settings_dict

    @staticmethod
    def _create_arff_dense(xs, ys, learning_task, file_handle):
        if len(xs) != len(ys):
            raise ValueError(f"len(features) = {len(xs)} != {len(ys)} = len(targets)")
        for x, y in zip(xs, ys):
            if learning_task == "mlc":
                line = ",".join(str(c) for space in [x, y] for c in space)
            elif learning_task == "mcc":
                line = ",".join(str(c) for c in x)
                line = ",".join([line, y])
            print(line, file=file_handle)
            
    @staticmethod
    def _arff_header_row(i, is_feature, values):
        column_name = f"feature{i + 1}" if is_feature else f"target{i + 1}"
        return f"@attribute {column_name} {values}"

    def read_data(self, f_path, t_path):
        features = np.load(f_path)
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
        
        if self.apply_pca:
            store_path_data =    [f"{self.arffs_path}", 
                                  f"{self.method}",
                                  f"{self.config.dataset_name}_pca", 
                                  f"{self.model_name}"]  
        else:
            
            store_path_data =    [f"{self.arffs_path}", 
                                  f"{self.method}",
                                  f"{self.config.dataset_name}", 
                                  f"{self.model_name}"] 
            
        store_path_settings = [f"{self.arffs_path}", 
                              f"{self.method}",
                              f"{self.config.dataset_name}", 
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
        
        with open(store_path_settings + "/features.s", 'w') as f: 
            for key, value in self.settings_dict.items(): 
                print('\n%s' % key, file=f)
                for v in value:
                    print('%s' % v, file=f)
        
        for split in splits:
            file_name = "/".join([store_path_data, f"{split}_features.arff"])     
            if os.path.exists(file_name):
                info_message("The path: {} exists...skipping. Creating only settings file.", file_name)
                continue
            
            else:
                with open(file_name, "w", newline='') as f:
                    print(f"@relation {self.config.dataset_name}", file=f)
                    for i in range(df['features'][split].shape[1]):        
                        print(self._arff_header_row(i, True, "numeric"), file=f)
                    
                    if self.learning_task == "mlc":
                        for j in range(df['targets'][split].shape[1]):
                            print(self._arff_header_row(i+1+j, False, "{0, 1}"), file=f)

                    elif self.learning_task == "mcc":
                        targ_string = set(np.array(sorted_alphanumeric(np.unique(df['targets'][split]))).astype(int))
                        print(self._arff_header_row(i+1, False, f"{targ_string}"), file=f)

                    print("@data", file=f)
                    self._create_arff_dense(df['features'][split], df['targets'][split], self.learning_task, f)

