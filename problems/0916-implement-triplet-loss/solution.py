import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    return F.relu(torch.pairwise_distance(anchor, positive) ** 2 - torch.pairwise_distance(anchor, negative) ** 2 + margin).mean()
