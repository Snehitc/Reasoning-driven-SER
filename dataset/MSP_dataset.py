# Author: Snehit
# E-mail: snehitc@gmail.com

import os
import librosa
import random
import numpy as np
import torch
import torchaudio.functional as F


class MSP_Dataset(torch.utils.data.Dataset):
    def __init__(self, df_speech, speech_dir, sr, norm_param, noise_dir_classes, SNRs):
        self.df_speech = df_speech.reset_index(drop=True)
        self.speech_dir = speech_dir
        self.sr = sr
        self.noise_dir = noise_dir_classes['noise_dir']
        self.noise_classes = noise_dir_classes['noise_classes']
        self.SNRs = SNRs
        self.mean, self.std = norm_param['wavs']['Mean'], norm_param['wavs']['Std']
        self.Min, self.Max = norm_param['labels']['Min'], norm_param['labels']['Max']
        
    def __len__(self):
        return len(self.df_speech)

    def extract_and_normalize_wav(self, wav_path):
        wav, _ = librosa.load(wav_path, sr=self.sr)
        target_len=self.sr*10   # 10 seconds
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
        wav = (wav - self.mean) / (self.std + 1e-6)
        return torch.tensor(wav, dtype=torch.float32)

    def mix_with_noise(self, wav):
        scene = random.choice(self.noise_classes)
        noise_files = os.listdir(os.path.join(self.noise_dir, scene))
        selected_file = random.choice(noise_files)

        noise_wav = self.extract_and_normalize_wav(os.path.join(self.noise_dir, scene, selected_file))
        snr_val = random.choice(self.SNRs)
        mixed = F.add_noise(wav.unsqueeze(0), noise_wav.unsqueeze(0), torch.tensor([snr_val]))
        return mixed.squeeze(0), snr_val, scene, noise_wav, selected_file
    
    
    def __getitem__(self, idx):
        row = self.df_speech.iloc[idx]
        wav_path = os.path.join(self.speech_dir, row["FileName"])
        clean_wav = self.extract_and_normalize_wav(wav_path)
        mixed_wav, snr_val, scene, noise_wav, selected_file = self.mix_with_noise(clean_wav)

        labels = torch.tensor([row["EmoAct"], row["EmoVal"], row["EmoDom"]], dtype=torch.float32)
        labels = (labels - self.Min) / (self.Max - self.Min)
        
        
        return {
            'audios': mixed_wav,
            'speech_file': row["FileName"],
            'noise_file': os.path.join(scene, selected_file),
            'snr_val': snr_val,
            'labels': labels,
        }