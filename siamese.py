import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """Siamese network for feature embedding"""
    def __init__(self, input_dim=512, embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward_one(self, x):
        output = self.embedding(x)
        # L2 normalize using functional API
        return F.normalize(output, p=2, dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2