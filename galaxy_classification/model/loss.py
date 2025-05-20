import torch.nn as nn
import torch

CLASS_GROUPS = {
    'Class 1': ['Class1.1', 'Class1.2', 'Class1.3'],
    'Class 2': ['Class2.1', 'Class2.2'],
    'Class 7': ['Class7.1', 'Class7.2', 'Class7.3'],
}

def regression_loss(outputs, labels, weights):
    mse = nn.MSELoss(reduction='none')
    losses = []
    for key in outputs.keys():
        pred, targ = outputs[key], labels[key]         # [B, Ck]
        se = mse(pred, targ)                           # [B, Ck]
        # compute the weights for each class
        w = torch.tensor([weights[c] for c in CLASS_GROUPS[f'Class {key[-1]}']], 
                         device=se.device, dtype=se.dtype)
        se = se * w.unsqueeze(0)
        losses.append(se.mean())
    return sum(losses)