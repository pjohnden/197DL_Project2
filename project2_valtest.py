# %%
import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType

# add. for KWS
import torchaudio, torchvision
import os
import matplotlib.pyplot as plt 
import librosa
import argparse
import numpy as np

from torchaudio.datasets import SPEECHCOMMANDS

print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)

import random

# add. for validation run
#import torchmetrics
from torchmetrics import Accuracy
import time

# %%
## FOR VALIDATION RUN
start_time = time.time()

#device = "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SETUP DATASET
class args:
    path = os.path.join('data','speech_commands')

CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    
# make a dictionary from CLASSES to integers
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
print(CLASS_TO_IDX)

if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

kws_test_dataset = torchaudio.datasets.SPEECHCOMMANDS(args.path, download=True, subset='testing')


# INSTANTIATE MODEL
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# FOR PREDICTION
list_length = kws_test_dataset.__len__()
targets = []
predictions = []
correct_count = 0
accounted_count = 0
error_count = 0
for index in range(list_length):
    try:
        audio_rel_path, sample_rate , label, _, _ = kws_test_dataset.get_metadata(index)
        audio_path = os.path.join(args.path, 'SpeechCommands', audio_rel_path )
        #print('audio path: ', audio_path)
        #print('label: ', label)
        targets.append(CLASS_TO_IDX[label])
        # Load data
        inputs = {ModalityType.TEXT: data.load_and_transform_text(CLASSES, device), 
                  ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], device, clip_duration=1, target_length=204,sample_rate=sample_rate,num_mel_bins=128), #default target length=204
        } 

        with torch.no_grad():
            embeddings = model(inputs)

        text_probs = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
        index = np.argmax(text_probs.cpu().numpy())
        predictions.append(index)

        if CLASS_TO_IDX[label]==index:
            correct_count += 1
    except:
        error_count += 1
        continue
    accounted_count += 1
    #print("Audio x Text: ", torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1))
    #print("Label:", CLASSES[index])

#print('targets: ', targets)
#print('predictions: ', predictions)

# Save a Copy of the Test Run Data
test_run_data = [targets, predictions]
with open('test_run_data.txt', 'w') as f:
    for sublist in test_run_data:
        line = ' '.join([str(item) for item in sublist])
        f.write(line + '\n')

# Compute Accuracy
targets = torch.tensor(targets)
predictions = torch.tensor(predictions)
accuracy = Accuracy('multiclass',num_classes=len(CLASSES))
accuracy(predictions, targets)
IMB_accuracy = accuracy.compute()

end_time = time.time()
execution_time = end_time - start_time

print("Run Summary:")
print(f'  accuracy: {IMB_accuracy}')
print(f'  correct predictions: {correct_count}')
print(f'  number of datapoints: {accounted_count}')
print(f'  unaccounted data(error):{error_count}')  
print(f'  Manual accuracy calculation: {correct_count/accounted_count}' )
print(f"  Execution time: {execution_time} seconds")



