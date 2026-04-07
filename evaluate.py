# Author: Snehit
# E-mail: snehitc@gmail.com

## Imports ##
import os
import yaml
import pandas as pd
import torch
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader

from dataset.MSP_dataset import MSP_Dataset
from model.model import WavLM_CLAP
from utils.utils import CCC_loss
from utils.utils import Get_Mellow_Captions


## Dataset Loader ##
def get_dataloader(config):
    dataset_path = config['speech']['speech_path']
    speech_audio_dir = os.path.join(dataset_path, config['speech']['audio_dir'])
    label_path = os.path.join(dataset_path, config['speech']['label_path'])
    norm_param = config['speech']['normalization']
    sr = config['speech']['sampling_rate']
    noise_dir_classes = config['noise']['test']
    SNRs = config["noise"]["test"]["SNRs"][0]

    df_labels = pd.read_csv(label_path)
    grouped = df_labels.groupby(df_labels.Split_Set)
    df_Test1 = grouped.get_group("Test1")
    df_Test1.reset_index(drop=True, inplace=True)

    MSP_Dataset_Test = MSP_Dataset(df_Test1, 
                                   speech_audio_dir, sr, norm_param,
                                   noise_dir_classes, SNRs)
    
    test_loader = DataLoader(MSP_Dataset_Test, 
                            batch_size=config['Dataloader']['batch_size'], 
                            shuffle=config['Dataloader']['shuffle'],)
    return test_loader



## SER Model Setup ##
def setup_model(config, device):
    model_config = config['models']
    model = WavLM_CLAP(model_config, device=device)

    model_ckpt = config['models']['ckpt']
    Model_PATH = os.path.join(config['models']['path'], model_ckpt)
    missing, unexpected = model.load_state_dict(torch.load(Model_PATH, map_location=torch.device(device), weights_only=True), strict=False)
    assert all(['CLAPText' in k for k in missing]), 'Warning: Missing CLAPText and *other* [Check 1. model or 2. state dict]'
    assert not unexpected, 'Warning: Unexpected keys found [Check 1. model and 2. state dict for missmatch]'

    return model



## Display: 1. Summary  2. Results ##
def Display_summary(config, model, test_loader, device):
    Dispay = [["Status", "Testing"],
              ["Device", device],
              ['Dataset', type(test_loader.dataset).__name__],
              ["Model (SER)", model.__class__.__name__],
              ["Model (Context)", "Mellow (Reasoning)"],
              ["SER Checkpoint", config['models']['ckpt']]]
    print("\n------ Setup Summary ------")
    print(tabulate(Dispay, tablefmt="grid"), '\n')

def Display_results(config, ccc):
    Dispay = [["SNR", f'SNR: {config["noise"]["test"]["SNRs"][0]}dB'],
              ["Metric", "CCC"],
              ['Arousal', ccc[0]],
              ["Valence", ccc[1]],
              ["Dominance", ccc[2]]]
    print("\n------ Results ------")
    print(tabulate(Dispay, tablefmt="grid"), '\n')



## Evaluate function:: 1. Context: Mellow Captions 2. SER: WavLM-CLAP ##
def evaluate(config, model, test_loader, device):
    Display_summary(config, model, test_loader, device)
    model.to(device)
    model.eval()
    labels_fold = []
    outputs_fold = []
    print("------ Evaluating... ------")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            Audio_Tensors = batch['audios']
            labels = batch['labels']
            Audio_Tensors = Audio_Tensors.to(device)
            
            # Audio Resoning: Generating Captions
            Text_Reasoning = Get_Mellow_Captions(Audio_Tensors, config['speech']['sampling_rate'], device)
            
            outputs = model(Audio_Tensors, Text_Reasoning)

            labels_fold.append(labels)
            outputs_fold.append(outputs)
    labels_all = torch.cat(labels_fold, 0)
    outputs_all = torch.cat(outputs_fold, 0)

    ccc = CCC_loss(outputs_all.detach().cpu(), labels_all)
    ccc = ccc.detach().cpu().numpy()
    Display_results(config, ccc)



## Run Evaluation ##
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    test_loader = get_dataloader(config)
    model = setup_model(config, device)
    evaluate(config, model, test_loader, device)    