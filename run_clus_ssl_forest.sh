#!/bin/bash

echo "dir: $1"

datasets=("Ankara")
methods=("RForest")
model_types=("effnet_b2")
seeds=("0" "1" "42")
n_labeled=("1" "5" "10" "25")
learning_settings=("ssl")

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
                  cd /home/ms3733/mlc/$1/$method/$dataset/$mt/seed_$seed/labeled_$N/$ls
                  echo $PWD
	          java -jar -Xmx64G ../../../../../../../../Clus_plus/Clus.jar -ssl -forest features.s 2>> log.txt &
               done
	    done
         done
      done
   done
done

