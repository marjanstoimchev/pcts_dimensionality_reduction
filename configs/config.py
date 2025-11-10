import argparse

class BaseConfig(object):
    num_workers = 4
    fold_num = 5
    seed = 42
    base_dir = '../rs_datasets/'
    learning_rate = 1e-4 #1e-3
    gamma = 0.1
    weight_decay = 1e-6

    optim = 'Adam'
    crit = "CE"
    sched = 'CosineAnnealingWarmRestarts' #step
    step_size = 10 # 25
    n_accumulate = 1 #5
    verbose_step = 1
    early_stopping = 10
    min_lr = 1e-5
    T_max = 500
    T_0 = 3
    

####################### Configs for MCC datasets ##############################
    
class OPTIMAL_31(BaseConfig):
    extension = 'jpg'
    n_classes = 31
    image_size = 256

    dataset_name = "OPTIMAL-31"
     
    out_features = {"Vgg16": 17,
                        "Vgg19": 16,
                        "resnet34": 14,
                        "resnet50": 12,
                        "resnet152": 12,
                        "effnet_b0": 14,
                        "effnet_b1": 15,
                        "effnet_b2": 16,
                        }
    
class RESISC45(BaseConfig):
    extension = 'jpg'
    n_classes = 45
    image_size = 256
    
    dataset_name = "RESISC45"
    
    out_features = {"Vgg16": 17,
                        "Vgg19": 17,
                        "resnet34": 17,
                        "resnet50": 15,
                        "resnet152": 15,
                        "effnet_b0": 22,
                        "effnet_b1": 24,
                        "effnet_b2": 24,
                        }

    
class RSSCN7(BaseConfig):
    extension = 'jpg'
    n_classes = 7
    image_size = 400
    
    dataset_name = "RSSCN7"
    
    out_features = {"Vgg16": 7,
                        "Vgg19": 6,
                        "resnet34": 5,
                        "resnet50": 4,
                        "resnet152": 4,
                        "effnet_b0": 5,
                        "effnet_b1": 5,
                        "effnet_b2": 5,
                        }
   
class UCM_mcc(BaseConfig):
    extension = 'tif'
    n_classes = 21
    image_size = 256
    
    dataset_name = "UCM"
    
    out_features = {"Vgg16": 12,
                        "Vgg19": 12,
                        "resnet34": 11,
                        "resnet50": 10,
                        "resnet152": 10,
                        "effnet_b0": 11,
                        "effnet_b1": 12,
                        "effnet_b2": 12,
                        }

class AID_mcc(BaseConfig):
    extension = 'jpg'
    n_classes = 30
    image_size = 600
    
    dataset_name = "AID"
    
    out_features = {"Vgg16": 18,
                        "Vgg19": 15,
                        "resnet34": 13,
                        "resnet50": 12,
                        "resnet152": 12,
                        "effnet_b0": 14,
                        "effnet_b1": 15,
                        "effnet_b2": 16,
                        }
    
####################### Configs for MLC datasets ##############################

class UCM_mlc(BaseConfig):
    extension = 'tif'
    n_classes = 17
    image_size = 256
    
    dataset_name = "UCM"
    
    out_features = {"Vgg16": 6,
                        "Vgg19": 3,
                        "resnet34": 8,
                        "resnet50": 7,
                        "resnet152": 7,
                        "effnet_b0": 11,
                        "effnet_b1": 12,
                        "effnet_b2": 12,
                        }

class AID_mlc(BaseConfig):
    extension = 'jpg'
    n_classes = 17
    image_size = 600
    
    dataset_name = "AID"
    
    out_features = {"Vgg16": 3,
                        "Vgg19": 5,
                        "resnet34": 8,
                        "resnet50": 7,
                        "resnet152": 7,
                        "effnet_b0": 9,
                        "effnet_b1": 10,
                        "effnet_b2": 11,
                        }


class MLRSNetConfig(BaseConfig):
    extension = 'jpg'
    n_classes = 60
    image_size = 256
    
    dataset_name = "MLRSNet_5_percent"
    
    out_features = {"Vgg16": 100,
                        "Vgg19": 78,
                        "resnet34": 12,
                        "resnet50": 2,
                        "resnet152": 3,
                        "effnet_b0": 33,
                        "effnet_b1": 33,
                        "effnet_b2": 34,
                        }

class DFC15(BaseConfig):
    extension = 'png'
    n_classes = 8
    image_size = 600
    
    dataset_name = "DFC_15"
    
    out_features = {"Vgg16": 1,
                        "Vgg19": 1,
                        "resnet34": 4,
                        "resnet50": 4,
                        "resnet152": 4,
                        "effnet_b0": 5,
                        "effnet_b1": 5,
                        "effnet_b2": 5,
                        }
    

class Ankara(BaseConfig):
    extension = 'bmp'
    extension_ms = '.mat'
    n_classes = 29
    image_size = 63

    dataset_name = "Ankara"
    
    out_features = {"Vgg16": 1,
                        "Vgg19": 1,
                        "resnet34": 10,
                        "resnet50": 10,
                        "resnet152": 10,
                        "effnet_b0": 20,
                        "effnet_b1": 32,
                        "effnet_b2": 22,
                        }

class ConfigSelector(BaseConfig):
    def __init__(self):
        super(ConfigSelector, self).__init__()
        self.args = self.ConfigParser()
        self.epochs = self.args.n_epochs
        self.batch_size = self.args.batch_size
        
    def select(self, dataset):
        if dataset == 'OPTIMAL-31':
            config = OPTIMAL_31()
        elif dataset == 'RESISC45':
            config = RESISC45()
        elif dataset == 'RSSCN7':
            config = RSSCN7()
        elif dataset == 'UCM_mcc':
            config = UCM_mcc()
        elif dataset == 'AID_mcc':
            config = AID_mcc()
            
        elif dataset == "MLRSNet_5_percent":
            config = MLRSNetConfig()
        elif dataset == "UCM_mlc":
            config = UCM_mlc()
        elif dataset == "AID_mlc":
            config = AID_mlc()
        elif dataset == "DFC_15":
            config = DFC15()
        elif dataset == "Ankara":
            config = Ankara()
        return config
    
    def ConfigParser(self):
        parser = argparse.ArgumentParser(description='Process hyper-parameters')
        parser.add_argument('--mode', type=str, default="train", help='Mode')
        parser.add_argument('--model_type', type=str, default="Vgg", help='Architecture')
        parser.add_argument('--loss', type=str, default='BCE', help='Type of loss function')
        parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size') # 64
        parser.add_argument('--seed', type=int, default=42, help='Seed')
        parser.add_argument('--lr', type=int, default=1e-4, help='Learning Rate')
        parser.add_argument('--features_type', type=str, default="FineTuned", help='Save path')
        args = parser.parse_args()
        return args
