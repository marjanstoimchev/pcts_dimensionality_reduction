from utils import *
from configs import *
from other_imports import *
from clus_utils.CreateArff import *
#% Create Arffs for CLUS


def main():

    model_types = [
        # "Vgg16", 
        # "Vgg19", 
        # "resnet34", "resnet50", 
        # "resnet152", "effnet_b0",
        # "effnet_b1", 
        "effnet_b2"]
    
    datasets = ["AID_mlc"]
    splits = ["train", "val", "test"]
    methods = ["PCT", "RForest"]
    learning_settings = ["sl", "ssl"]
    seeds = ["0"]
    learning_task = "mlc"
    percentage_labeled_examples = ["25"] 
    apply_pca = False
    str = "pca" if apply_pca else "original"
    root_dir = f"FEATURES/{str}"
    
    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                for model_type in model_types:   
                    for percentage_labeled in percentage_labeled_examples:
                        for learning_setting in learning_settings:
                              
                            features_path = f"{dataset}_features/{model_type}"
                            features_paths = []
                            targets_paths = []
                            
                            for split in splits:
                                features_paths += ["/".join([root_dir, features_path, f"{split}_features_{model_type}.npy"])]
                                targets_paths += ["/".join([root_dir, features_path, f"{split}_targets.npy"])]
                            
                            cs =  CreateSettingsSSL(f"Arfs_{learning_task}",
                                                    features_paths,
                                                    targets_paths,
                                                    dataset, 
                                                    model_type, 
                                                    percentage_labeled, 
                                                    method,
                                                    learning_setting,
                                                    learning_task,
                                                    apply_pca,
                                                    seed)          

                            info_message("Dataset: {}, seed: {}, model: {}", dataset, seed, model_type)
                            cs.create()

if __name__ == "__main__":
    main()
