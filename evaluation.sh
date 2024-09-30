#!/bin/bash

file_names=("ldm.csv" "adm.csv" "ddpm.csv" "iddpm.csv" "pndm.csv" "sd.csv" "glide.csv") #"styleGAN.csv" "Diff-StyleGAN2.csv" "Diff-ProjectedGAN.csv" "ProGAN.csv" "ProjectedGAN.csv"
model_names=("ldm" "adm" "ddpm" "iddpm" "pndm" "stablediffusion" "glide")

for file_name in "${file_names[@]}"; do
  for model_name in "${model_names[@]}"; do
    python ./blip2_test.py --model_path "./BLIP2/${model_name}FineTune/" --batchSize 1 --dataset "./Test/${file_name}" --save_output "./Results/BLIP2_$(echo $model_name | tr '[:lower:]' '[:upper:]')_${file_name}"
  done
done

# #!/bin/bash

# file_names=("ldm.csv" "adm.csv" "ddpm.csv" "iddpm.csv" "pndm.csv" "sd.csv" "glide.csv") # "styleGAN.csv" "Diff-StyleGAN2.csv" "Diff-ProjectedGAN.csv" "ProGAN.csv" "ProjectedGAN.csv"
# model_names=("ldm") #"ldm" "adm" "ddpm" "iddpm" "pndm" "stablediffusion" "glide"

# for file_name in "${file_names[@]}"; do
#   for model_name in "${model_names[@]}"; do
#     python ./VitGPT2_test.py --model_path "./Vit-GPT2/${model_name}FineTune/" --batchSize 8 --dataset "./BLIP2/${file_name}" --save_output "./ViTGPT2/ViTGPT2_$(echo $model_name | tr '[:lower:]' '[:upper:]')_${file_name}"
#   done
# done
