#!/bin/bash

echo "dir: $1"

datasets=("OPTIMAL-31_pca")
methods=("RForest")
model_types=("resnet152" "effnet_b2")
seeds=("0" "1" "42")
n_labeled=("1" "5" "10" "25")
learning_settings=("sl" "ssl")
variances=("exp_variance_60" "exp_variance_80" "exp_variance_95")

for dataset in ${datasets[@]}
do
   for method in ${methods[@]}
   do
      for mt in ${model_types[@]}
      do
        for variance in ${variances[@]}	
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
                  cd /home/ms3733/mlc/$1/$method/$dataset/$variance/$mt/seed_$seed/labeled_$N/$ls
                  echo $PWD
	          java -jar -Xmx128G ../../../../../../../../../Clus_plus/Clus.jar -ssl -forest features.s 2>> log.txt &
               done
	    done
           done
         done
      done
   done
done

