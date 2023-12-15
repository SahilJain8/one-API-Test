from intel_extension_for_transformers.transformers.pipeline import pipeline
import time
import torch
import pandas as pd
from tqdm import tqdm
df = pd.read_csv("FashionV2.csv")

column = df['img'].values
time_per_step = []

complete_start_time = time.time()
text_classifier = pipeline(
    task="image-to-text",
     model="Salesforce/blip-image-captioning-large",
    framework="pt",
    device=torch.device("cpu"),
)
for data in tqdm(column[:1000]):
    start = time.time()
    a = text_classifier(data)
    # print(a)
    end = time.time()
    time_per_step.append(end - start)

complete_end = time.time()
complete_time_taken = complete_end - complete_start_time
print(complete_time_taken)
