from transformers import pipeline
import time
import torch
import pandas as pd
from tqdm import tqdm
df = pd.read_csv("FashionV2.csv")

column = df['img'].values


complete_start_time_cpu= time.time()
text_classifier = pipeline(
    task="image-to-text",
     model="Salesforce/blip-image-captioning-large",
    framework="pt",
    device=torch.device("cpu"),
)
for data in tqdm(column[:1000]):
    a = text_classifier(data)



complete_end_cpu = time.time()
complete_time_taken_cpu = complete_end - complete_start_time
print(complete_time_taken_cpu)
