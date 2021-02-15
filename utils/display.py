import torch
import tqdm
import magicpath
import numpy as np

from models import load_model
from data_loader import setup_dataset
assert magicpath

run_dir = 'runs/hyper_params_step_20/2020-05-24_16h30min'
split = 'test'
device = 'cuda:0'

model, config = load_model(run_dir, metric='accuracy')
config.training.batch_size = 16
config.runtime.num_workers = 6

train_dataloader, val_dataloader = setup_dataset(config)
if split == 'test':
    dataloader = setup_dataset(config, test_only=True)

elif split == 'val':
    dataloader = val_dataloader

elif split == 'train':
    dataloader = train_dataloader

model = model.to(device)
model.eval()
results = []
family_probas = []
genus_probas = []
species_probas = []
predicted_probas = []
correct = []
with torch.no_grad():
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader)):
        data = data.to(device)
        output = model(data)
        label = (target['label']).to(device)

        normalization = torch.sum(torch.exp(output), axis=-1)

        same_family = target["same_family"].to(device)
        family_exp = torch.sum(torch.exp(output) * same_family, axis=-1)
        family_proba = family_exp / normalization
        family_probas += family_proba.cpu().numpy().tolist()

        same_genus = target["same_genus"].to(device)
        genus_exp = torch.sum(torch.exp(output) * same_genus, axis=-1)
        genus_proba = genus_exp / normalization
        genus_probas += genus_proba.cpu().numpy().tolist()

        species_proba = torch.exp(output[torch.arange(label.shape[0]),
                                         label])
        species_proba = species_proba / normalization
        species_probas += species_proba.cpu().numpy().tolist()
        max_output = torch.max(output, dim=-1).values
        predicted_proba = torch.exp(max_output) / normalization
        predicted_probas += predicted_proba.cpu().numpy().tolist()
        class_pred = torch.argmax(output, dim=-1)
        correct += (label == class_pred).cpu().numpy().tolist()


metrics = {}
metrics['accuracy'] = np.mean(correct)
metrics['species_probas'] = np.mean(species_probas)
metrics['genus_probas'] = np.mean(genus_probas)
metrics['family_probas'] = np.mean(family_probas)
metrics['incorrect pred proba species'] = np.mean(
    (1 - np.asarray(correct)) * np.asarray(predicted_probas))
metrics['incorrect target proba species'] = np.mean(
    (1 - np.asarray(correct)) * np.asarray(species_probas))
metrics['incorrect target proba genus'] = np.mean(
    (1 - np.asarray(correct)) * np.asarray(genus_probas))
metrics['incorrect target proba family'] = np.mean(
    (1 - np.asarray(correct)) * np.asarray(family_probas))

print(metrics)
