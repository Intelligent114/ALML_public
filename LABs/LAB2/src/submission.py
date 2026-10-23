"""LAB2: edit and submit only this file. NumPy + PyTorch are allowed.
No file/network access, pre-trained weights, target-function encoding, or search.
All predictions/labels have shape (n,1). NumPy parameters are W1,b1,W2,b2,W3,b3:
W1=(input_dim,h1), W2=(h1,h2), W3=(h2,1), biases are one-dimensional.
NumPy routines must NOT use torch/autograd. Inputs must not be modified.
"""
import numpy as np
import torch
from torch import nn

def get_config():
    """Replace with selected fixed values from the 8 validation trials.
    scale: .25,1,4; lr: .001,.003; lam: 0 or one of the two positive extras.
    This function adds no separate code points.
    """
    return {'scale': 1.0, 'lr': 0.001, 'lam': 0.0}

def fourier_features(x, scale):
    """x: floating numpy ndarray or torch tensor (n,d); scale positive.
    Return same backend/dtype/device (n,13*d). Per coordinate q, concatenate
    [x_q, sin(pi*scale*2**j*x_q) for j=0..5, cos(...) for j=0..5].
    Tensor version must preserve autograd. No random frequencies.
    """
    raise NotImplementedError

def numpy_forward(x, params):
    """Two ReLU hidden layers + linear output. Return (pred, cache).
    cache is your own object consumed only by numpy_loss_backward.
    Support arbitrary positive n/input_dim/h1/h2, float32/float64.
    """
    raise NotImplementedError

def numpy_loss_backward(pred, y, cache, params, lam):
    """Return (objective float, gradients dict with the same keys/shapes).
    J=mean((pred-y)**2)+lam/2*(||W1||_F**2+||W2||_F**2+||W3||_F**2).
    No penalty for biases; ReLU derivative at zero is defined as zero.
    """
    raise NotImplementedError

def build_torch_model(params):
    """Return nn.Sequential(Linear,ReLU,Linear,ReLU,Linear).
    Match widths from supplied arrays, transpose W into Linear.weight.
    Load ALL supplied parameters exactly; dtype must follow params.
    Returned model starts on CPU. Do not change architecture/initialize anew.
    """
    raise NotImplementedError

def torch_objective(pred, y, model, lam):
    """Return differentiable scalar Tensor equal to the NumPy objective."""
    raise NotImplementedError
