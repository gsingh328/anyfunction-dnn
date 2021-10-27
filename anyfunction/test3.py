import torch
import torch.nn as nn


encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
src = torch.rand(10, 32, 512)
out = encoder_layer(src)

print(src.shape)
print(out.shape)
