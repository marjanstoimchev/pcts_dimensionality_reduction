#!/bin/bash

echo "dir: $1"

datasets=("RESISC45")
methods=("PCT")
model_types=("Vgg19")
seeds=("111")
n_labeled=("25")
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
						java -jar -Xmx16G ../../../../../../../../Clus_plus/Clus.jar -ssl features.s 2>> log.txt &
					done
				done
			done
		done
	done
done

