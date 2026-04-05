# Author: Snehit
# E-mail: snehitc@gmail.com

import torch
import sys
sys.path.append("./mellow")
from mellow import MellowWrapper


def CCC_loss(pred, lab, m_lab=None, v_lab=None, is_numpy=False):
    # Concordance Correlation Coefficient (CCC) loss
    # Taken from "Speech Emotion Recognition in Naturalistic Conditions Challenge"
    # Reference: https://lab-msp.com/MSP-Podcast_Competition/IS2025/
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    if is_numpy:
        pred = torch.Tensor(pred).float().cuda()
        lab = torch.Tensor(lab).float().cuda()
    
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
    return ccc



# setup mellow
mellow = MellowWrapper(
                    config = "v0",
                    model = "v0",
                    device=0,
                    use_cuda=True,
                )


def Get_Mellow_Captions(audio_tensors, sr, device):
    # Audio Resoning: Generating Captions from Mellow (at default settings)
    prompts = ["caption the audio."] * len(audio_tensors)
    mellow_inputs = [[a.unsqueeze(0), p] for a, p in zip(audio_tensors.to(device), prompts)]
    
    response = mellow.Custom_generate(examples=mellow_inputs, current_sr=sr, 
                                        max_len=300, top_p=0.8, temperature=1.0)
    return response