#!/bin/bash

echo "dir: $1"
echo "ensemble: $2"

datasets=("MLRSNet")
feature_types=("Transfer" "FineTune")
model_types=("Vgg16" "Vgg19" "resnet34" "resnet50" "resnet152" "effnet_b0" "effnet_b1" "effnet_b2")

for dataset in ${datasets[@]}
do
   for ft in ${feature_types[@]}
   do
      for mt in ${model_types[@]}
      do
         printf '\n\n\n\n\n'
	 echo "---------------------------------------------------------"
         echo "Dataset: $dataset , feature type: $ft , model type: $mt"
	 echo "---------------------------------------------------------"
         cd /home/ms3733/mlc/$1/$2/$dataset/$ft/$mt/full_split
         echo $PWD
	 java -jar -Xmx256G ../../../../../../../Clus_plus/Clus.jar -forest features.s
      done
   done
done


