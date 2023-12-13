import math
import torch
import torch.nn as nn
from torch.nn import functional as F

"""Attribution (followed this tutorial!): https://medium.com/@alexmriggio/lora-low-rank-adaptation-from-scratch-code-and-theory-f31509106650 
"""

class LinearLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, lora_alpha=32, lora_dropout=0.0):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        assert rank > 0, "Variable 'rank' is not greater than zero. Choose a rank of 1 or greater."

        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False
        
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.constant_(self.lora_B.weight, 0)

        self.scale = self.lora_alpha / self.rank
    
    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scale

        return pretrained_out + lora_out

def freeze_model(model):
    """Freeze all layers except LoRA modules"""
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

def create_lora(module, rank, lora_dropout, lora_alpha):
    """Converts a linear module to a LoRA linear module"""
    k, d = module.weight.shape
    lora = LinearLoRA(d, k, rank=rank, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        if module.bias is not None:
            lora.pretrained.bias.copy_(module.bias)
    return lora

def unfreeze_model(model):
    """Unfreezes all parameters in a model by setting requires_grad to True."""
    for _, param in model.named_parameters():
        param.requires_grad = True

        
def create_linear(module):
    """Converts a LoRA linear module back to a linear module."""
    k, d = module.pretrained.weight.shape
    linear = nn.Linear(d, k, bias=True)
    
    with torch.no_grad():
        linear.weight.copy_(module.pretrained.weight + (module.lora_B.weight @ module.lora_A.weight) * module.scale)
        linear.bias.copy_(module.pretrained.bias)
        
    return linear

def add_lora_layers(model, rank=8, lora_alpha=16, lora_dropout=0.0):
    """Converts the model's linear layers to LoRA"""
    to_change = []
    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, nn.Dropout):
          module.p = 0.0

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            temp_lora = create_lora(module, rank=rank, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
            setattr(model, name, temp_lora)   
        else:
            add_lora_layers(module, rank, lora_alpha, lora_dropout)
    
def merge_lora_layers(model, dropout=0.0):
    """Return the linear LoRA layers back to linear"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
          module.p = 0.0

    for name, module in model.named_children():
        if isinstance(module, LinearLoRA):
            temp_linear = create_linear(module)
            setattr(model, name, temp_linear)
        else:
            merge_lora_layers(module, dropout=0.0)