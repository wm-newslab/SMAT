"""Gated-attention MIL model, checkpoint restoration, and inference."""

import numpy as np
import torch
from torch import nn

from src.bags import make_loader


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim, class_count, hidden_dim=64, attention_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, class_count)

    def forward(self, instances, mask):
        encoded = self.encoder(instances)
        gated = torch.tanh(self.attention_v(encoded)) * torch.sigmoid(
            self.attention_u(encoded)
        )
        scores = self.attention_w(gated).squeeze(-1)
        attention = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        bag = torch.sum(attention.unsqueeze(-1) * encoded, dim=1)
        return self.classifier(bag), attention


def infer(
    model,
    features,
    offsets,
    labels,
    users,
    batch_size,
    device,
    excluded=None,
):
    predictions, attentions = [], []
    model.eval()
    loader = make_loader(
        features, offsets, labels, users, batch_size, excluded=excluded
    )
    with torch.no_grad():
        for instances, mask, _ in loader:
            logits, attention = model(instances.to(device), mask.to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            attention = attention.cpu().numpy()
            mask = mask.numpy()
            for row_index in range(len(attention)):
                attentions.append(attention[row_index, : int(mask[row_index].sum())])
    return np.asarray(predictions), attentions


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_model(checkpoint, device):
    model = GatedAttentionMIL(
        checkpoint["input_dim"],
        len(checkpoint["classes"]),
        checkpoint["hidden_dim"],
        checkpoint["attention_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
