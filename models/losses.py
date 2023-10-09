import torch
import torch.nn as nn


class NTXentLoss(nn.Module):

    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits):
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0]).to(logits).long()
        loss = self.loss_fct(logits, labels)
        return loss
