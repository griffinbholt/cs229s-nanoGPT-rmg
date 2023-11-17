import torch

def quantize(X):
    """
    Given a weight matrix/input vector, quantize the weights
    Returns quantized weights (in torch.int8), scale, offset
    """
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    scale = x_range / 255
    offset = (-torch.min(X) / scale - 128).round()
    X_quant = torch.clip((X / scale + offset).round(), -128, 127)
    return X_quant, scale, offset

def dequantize(X_quant, scale, offset):
    """
    Dequantize a given output vector X_quant given a scale and offset
    """
    return (X_quant - offset) * scale 