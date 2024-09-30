#!/bin/bash

file_names=("pndm.csv" "sd.csv" "glide.csv") #"styleGAN.csv" "Diff-StyleGAN2.csv" "Diff-ProjectedGAN.csv" "ProGAN.csv" "ProjectedGAN.csv"
model_names=("adm" "ddpm" "iddpm" "pndm" "sd" "glide")

for file_name in "${file_names[@]}"; do
  for model_name in "${model_names[@]}"; do
    python ./blip2_test.py --model_path "./BLIP2/${model_name}FineTune/" --batchSize 32 --dataset "./BLIP2/${file_name}" --save_output "./Results/BLIP2$(echo $model_name | tr '[:lower:]' '[:upper:]')_${file_name}"
  done
done

# #!/bin/bash

# file_names=("ldm.csv" "adm.csv" "ddpm.csv" "iddpm.csv" "pndm.csv" "sd.csv" "glide.csv") # "styleGAN.csv" "Diff-StyleGAN2.csv" "Diff-ProjectedGAN.csv" "ProGAN.csv" "ProjectedGAN.csv"
# model_names=("ldm") #"ldm" "adm" "ddpm" "iddpm" "pndm" 

# for file_name in "${file_names[@]}"; do
#   for model_name in "${model_names[@]}"; do
#     python ./VitGPT2_test.py --model_path "./Vit-GPT2/${model_name}FineTune/" --batchSize 8 --dataset "./BLIP2/${file_name}" --save_output "./Degraded/ViTGPT2/ViTGPT2LR112$(echo $model_name | tr '[:lower:]' '[:upper:]')_${file_name}"
#     # python ../blip2_test.py --model_path "./${model_name}FineTune/" --batchSize 32 --dataset "./${file_name}" --save_output "../Degraded/ViTGPT2/ViTGPT2LR112$(echo $model_name | tr '[:lower:]' '[:upper:]')_${file_name}"
#   done
# done

# file_names=("COCO.csv" "flickr30k_224.csv" "SD2.csv" "SDXL.csv" "IF.csv" "DALLE2.csv" "SGXL.csv" "Control_COCO.csv" "lama_224.csv" "SD2Inpaint_224.csv" "lte_SR4_224.csv" "SD2SuperRes_SR4_224.csv" "deeperforensics_faceOnly.csv" "AdvAtk_Imagenet.csv" "Backdoor_Imagenet.csv" "DataPoison_Imagenet.csv")
# model_names=("SaveFineTuneAntifakePromptBlip2") 

# for file_name in "${file_names[@]}"; do
#   for model_name in "${model_names[@]}"; do
#     python blip2_test.py --model_path "../../ICPR2024/ICPR2024/${model_name}/" --batchSize 32 --dataset "./AntifakePromptDataset/${file_name}" --save_output "./ResultsFakePrompt/AntifakePrompt_test_${file_name}"
#   done
# done

# file_names=("progan.csv" "cyclegan.csv" "biggan.csv" "stylegan.csv" "gaugan.csv" "stargan.csv" "deepfake.csv" "seeingdark.csv" "san.csv" "crn.csv" "imle.csv" "guided.csv" "ldm_200.csv" "ldm_200_cfg.csv" "ldm_100.csv" "glide_100_27.csv" "glide_50_27.csv" "glide_100_10.csv" "dalle.csv")
# model_names=("UniversalFakeDetectTuning") 

# for file_name in "${file_names[@]}"; do
#   for model_name in "${model_names[@]}"; do
#     python blip2_test.py --model_path "${model_name}/" --batchSize 32 --dataset "./UniversalFakeDetect/test/${file_name}" --save_output "./UniversalFakeDetect/results/UniversalFakeDetectTuning_${file_name}"
#   done
# done
