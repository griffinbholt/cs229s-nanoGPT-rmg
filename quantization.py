import torch

def quantize(X):
    """
    Given a weight matrix/input vector, quantize the weights
    Returns quantized weights (in torch.int8), scale, offset
    """
    # Implementing the Q fn from the quadapter paper in comments
    # Q(x) = x * (clip(round(x/s + o), 0, 255) - o) 
    # where s = (max(x) - min(x)) / 255, o = round(-min(x)/s)

    # simple zero-point is here
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    scale = x_range / 255
    offset = (-torch.min(X) / scale - 128).round()
    X_quant = torch.clip((X / scale + offset).round(), -128, 127)

    # offset = - torch.min(X) / scale

    # X_quant = scale * (torch.clip((X / scale + offset).round(), 0, 255) - offset)
    return X_quant, scale, offset

def dequantize(X_quant, scale, offset):
    """
    Dequantize a given output vector X_quant given a scale and offset
    """
    return (X_quant - offset) * scale 
    # tbqh I feel like all the operations for quadapter Q cancel each other out, so not sure how to proceed?
    # return ((X_quant / scale) + offset - offset) * scale ???