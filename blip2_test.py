import os
import PIL
import pandas as pd
import numpy as np
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import matplotlib.pyplot as plt
import tqdm
import time
import argparse

# Set random seed for PyTorch
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for NumPy
np.random.seed(RANDOM_SEED)

# Custum Round Function
def multiple_custom_round(values):
    result = []
    for value in values:
        if value > 0.6:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)

# Map Text to Binary Value
def map_text_to_binary(text):
    if text == "fake":
        return 1
    elif text == "real":
        return 0
    else:
        return None  

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Fine-Tuned BLIP-2 for Diffusion-based Generated Images Detection.")
    parser.add_argument('--model_path', type=str, default='./SaveFineTune',
                        help='Path to the trained model.')
    parser.add_argument('--dataset', default='./data/test.csv', type=str,
                        help='Path to the testing CSV file')

    opt = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    config = PeftConfig.from_pretrained(opt.model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
    model = PeftModel.from_pretrained(model, opt.model_path)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")


    test_df = pd.read_csv(opt.dataset)
    test_dataset = ImageCaptioningDataset(test_df, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)


    result = []
    start_time = time.time()

    for batch in tqdm.tqdm(test_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
        
        generated_ids = model.generate(pixel_values=pixel_values, max_length=2)
        result.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    result_df = pd.DataFrame({
        'image': test_df['image'],
        'GT': test_df['text'], # GT : GroundTruth
        'Tlabel': test_df['label'], # TLabel: True Label (binary value : 0 or 1)
        'Pred': result,
        'Plabel': [map_text_to_binary(x) for x in result]
    })


    fpr, tpr, th = metrics.roc_curve(result_df['Tlabel'], result_df['Plabel'])
    auc = metrics.auc(fpr, tpr)
    preds = multiple_custom_round(np.asarray(result_df['Plabel']))
    accuracy = accuracy_score(result_df['Plabel'], result_df['Tlabel'])
    f1Score = f1_score(torch.tensor(result_df['Tlabel']),torch.tensor( result_df['Plabel']))

    print("METRICS")
    print("AUC: ", auc, "| Accuracy: ", accuracy, "| F1-Score: ", f1Score)
    print('', str(round(accuracy*100,2)), "|", str(round(auc*100,2)), "|", str(round(f1Score*100,2)))










