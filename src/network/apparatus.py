import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from src.encoder import get_encoder

# encoder = get_encoder(encoding='hashgrid', input_di=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19)

def positional_encoding(positions, freqs):
    # freq_bands: (1, 2, 4, ..., 2^(freqs-1))
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (..., F)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

# use Tensor to store the neural field features and decode with mlp (get Tensor features by mlp)
class Tensor_Only_Render(torch.nn.Module):
    def __init__(self, inChanel, pospe=6, num_layers=4, hidden_dim=32, skips=[2], out_dim=1, last_activation="sigmoid"):
        super(Tensor_Only_Render, self).__init__()
        self.pospe = pospe
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        # 方法1：position encoding
        # self.in_dim = (3 + 2 * pospe * 3) + inChanel
        self.in_dim = inChanel
        # Linear layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.in_dim, hidden_dim)] + [torch.nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                          else torch.nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers - 1, 1)])
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        # Activations
        self.activations = torch.nn.ModuleList([torch.nn.LeakyReLU() for i in range(0, num_layers - 1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(torch.nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(torch.nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, pts, features):
        # indata = [features, pts]
        indata = [features]
        # position encoding
        # if self.pospe > 0:
            # indata += [positional_encoding(pts, self.pospe)]
        sigma = torch.cat(indata, dim=-1)
        mlp_in = sigma[..., :self.in_dim]
        for i in range(len(self.layers)):
            linear = self.layers[i]
            activation = self.activations[i]
            if i in self.skips:
                sigma = torch.cat([mlp_in, sigma], -1)
            sigma = linear(sigma)
            sigma = activation(sigma)
        return sigma

# use Hash Table to store the neural field features and decode with mlp (same way as NAF use)
class Hash_Only_Render(torch.nn.Module):
    def __init__(self, encoder, bound=0.3, num_layers=4, hidden_dim=32, skips=[2], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        """ 需要把encoder放到类的里面，否则效果会变差，原因暂时未知 """
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        # Linear layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.in_dim, hidden_dim)] + [torch.nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                          else torch.nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers - 1, 1)])
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        # Activations
        self.activations = torch.nn.ModuleList([torch.nn.LeakyReLU() for i in range(0, num_layers - 1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(torch.nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(torch.nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        sigma = self.encoder(x, self.bound)
        input_pts = sigma[..., :self.in_dim]
        for i in range(len(self.layers)):
            linear = self.layers[i]
            activation = self.activations[i]
            if i in self.skips:
                sigma = torch.cat([input_pts, sigma], -1)
            sigma = linear(sigma)
            sigma = activation(sigma)
        return sigma

class PE_Only_Render(torch.nn.Module):
    def __init__(self, encoder, bound=0.3, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="relu"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        # 方法1：position encoding
        # self.pospe = pospe
        # self.in_dim = (3 + 2 * self.pospe * 3)
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        # Linear layers
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.in_dim, hidden_dim)] + [torch.nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                          else torch.nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers - 1, 1)])
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        # Activations
        self.activations = torch.nn.ModuleList([torch.nn.LeakyReLU() for i in range(0, num_layers - 1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(torch.nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(torch.nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        sigma = self.encoder(x, self.bound)
        input_pts = sigma[..., :self.in_dim]
        for i in range(len(self.layers)):
            linear = self.layers[i]
            activation = self.activations[i]
            if i in self.skips:
                sigma = torch.cat([input_pts, sigma], -1)
            sigma = linear(sigma)
            sigma = activation(sigma)
        return sigma

# use both Hash Table and Tensor to store the neural field features and decode with mlp (get Tensor features by mlp)
class Hash_Tensor_Render(torch.nn.Module):
    def __init__(self, encoder, inChanel, bound=0.3, num_layers=4, hidden_dim=32, skips=[], out_dim=1, last_activation="sigmoid"):
        super(Hash_Tensor_Render, self).__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        # hash encoding
        self.in_mlpC = self.in_dim + inChanel
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.in_mlpC, hidden_dim)] + [torch.nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                           else torch.nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers - 1, 1)])
        self.layers.append(torch.nn.Linear(hidden_dim, out_dim))
        # Activations
        self.activations = torch.nn.ModuleList([torch.nn.LeakyReLU() for i in range(0, num_layers - 1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(torch.nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(torch.nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, pts, features):
        # indata = [features, pts]
        indata = [features]
        # hash encoding
        mlp_in = self.encoder(pts, self.bound)
        indata += [mlp_in]
        sigma = torch.cat(indata, dim=-1)
        mlp_in = mlp_in[..., :self.in_dim]
        for i in range(len(self.layers)):
            linear = self.layers[i]
            activation = self.activations[i]
            if i in self.skips:
                sigma = torch.cat([mlp_in, sigma], -1)
            sigma = linear(sigma)
            sigma = activation(sigma)
        return sigma




