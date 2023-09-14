# Harnessing the Power of Large Vision Language Models for Synthetic Image Detection

This is the **official repository** for the [**paper**](https://arxiv.org/abs/) "*Harnessing the Power of Large Vision Language Models for Synthetic Image Detection*".

![assets/approach.png](assets/approach.png)

## Requirements
``` python
pip install -r requirements.txt
```

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

## :book: Citation
if you make use of our work, please cite our paper
```
@{

}
```
