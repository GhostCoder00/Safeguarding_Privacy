#source file: https://github.com/face-analysis/emonet/blob/master/emonet/evaluation.py

import numpy as np
import torch

# Test loop
def evaluate(net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str)-> dict:
    """
    Evaluate the network on the given dataloader.

    Args:
        net (torch.nn.Module): Network
        dataloader (torch.utils.data.DataLoader): Dataloader
        device (torch.device): Device to use

    Returns:
        out (dict): Dictionary with the results
    """
    for index, data in enumerate(dataloader):
        images = data.to(device)
        with torch.no_grad():
            out = net(images)    
        return out
