#!/bin/bash

echo "dir: $1"

datasets=("AID")
methods=("RForest" "PCT")
model_types=("Vgg16" "Vgg19" "resnet34" "resnet50" "resnet152" "effnet_b0" "effnet_b1" "effnet_b2")
seeds=("0" "1" "42")
n_labeled=("1" "5" "10" "25")
learning_settings=("sl" "ssl")

for dataset in ${datasets[@]}
do
   for method in ${methods[@]}
   do
      for mt in ${model_types[@]}
      do
         for seed in ${seeds[@]}
         do
            for N in ${n_labeled[@]}    	
	    do
               for ls in ${learning_settings[@]}
	       do
	          printf '\n\n\n\n\n\n\n'
	          echo "---------------------------------------------------------"
	          echo "----------------------Processing -------------------------"
	          echo "---------------------------------------------------------"
                  echo "DATASET: $dataset, METHOD: $method, MODEL TYPE: $mt, SEED: $seed , PERCENTAGE LABELED: $N, LEARNING SETTING: $ls"
	          echo "---------------------------------------------------------"
                  cd /home/ms3733/mlc/$1/$method/$dataset/FineTune/$mt/seed_$seed/labeled_$N/$ls
                  echo $PWD
	          java -jar -Xmx3G ../../../../../../../../../Clus_plus/Clus.jar -ssl features.s &
               done
	    done
         done
      done
   done
done

