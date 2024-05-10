# Harnessing the Power of Large Vision Language Models for Synthetic Image Detection

This is the **official repository** for the [**paper**]([https://arxiv.org/abs/](https://arxiv.org/pdf/2404.02726)) "*Harnessing the Power of Large Vision Language Models for Synthetic Image Detection*".

![assets/approach.png](assets/approach.png)

## Requirements
``` python
pip install -r requirements.txt
```

## SOTA Detection Methods

We use the codes of detection methods provided in the corresponding paper. 

- [AntifakePrompt](https://github.com/nctu-eva-lab/antifakeprompt)
- [DE-FAKE](https://github.com/zeyangsha/De-Fake)
- [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)
- [CNNDetection](https://github.com/PeterWang512/CNNDetection)
- [DIRE](https://github.com/ZhendongWang6/DIRE)
- [FusingGlobalandLocal](https://github.com/littlejuyan/FusingGlobalandLocal)
- [ClipBased-SyntheticImageDetection](https://grip-unina.github.io/ClipBased-SyntheticImageDetection/)
- [DMimageDetection](https://github.com/grip-unina/DMimageDetection)
- [Diffusers](https://github.com/davide-coccomini/Detecting-Images-Generated-by-Diffusers)

## Training (Optional)
This step can be skipped, and you can directly test the model in the following section with a pre-trained model.

To train your own model:
```python
python blip2_detect.py --dataset ./data/train.csv --epochs 20 --lr 5e-5 
```
## Evaluation
To run the evaluation, use the following command:
```python
python blip2_test.py --model_path ./SaveFineTune --dataset ./data/test.csv
```
## Performance
After training for 20 epochs, you will obtain accuracy and F1-score scores close to the percentages below:

```python
{'LDM' : 99.12/99.13, 'ADM' : 85.24/82.97, 'DDPM' : 98.47/98.47, 'IDDPM' : 97.02/96.97, 'PNDM' : 99.22/99.23, 'SD v1.4' 77.68/71.79: , 'GLIDE' : 97.09/97.05} 
```
## Dataset

The dataset used in this project is sourced from the work of [Towards the Detection of Diffusion Model Deepfakes](https://arxiv.org/abs/2210.14571), available at [Link to Original Dataset Repository](https://github.com/jonasricker/diffusion-model-deepfake-detection).


## :book: Citation
if you make use of our work, please cite our papers
```
@article{keita2024harnessing,
  title={Harnessing the Power of Large Vision Language Models for Synthetic Image Detection},
  author={Keita, Mamadou and Hamidouche, Wassim and Bougueffa, Hassen and Hadid, Abdenour and Taleb-Ahmed, Abdelmalik},
  journal={arXiv preprint arXiv:2404.02726},
  year={2024}
}
```
```
@article{keita2024bi,
  title={Bi-LORA: A Vision-Language Approach for Synthetic Image Detection},
  author={Keita, Mamadou and Hamidouche, Wassim and Eutamene, Hessen Bougueffa and Hadid, Abdenour and Taleb-Ahmed, Abdelmalik},
  journal={arXiv preprint arXiv:2404.01959},
  year={2024}
}
```
