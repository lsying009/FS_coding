import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# https://github.com/npvoid/neural_heterogeneity/blob/main/SuGD_code/layers.py


class SuSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 5.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, u, theta):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        input = u - theta
        ctx.save_for_backward(input)
        out = torch.gt(input, 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # SuSpike.scale 
        grad = grad_input / (SuSpike.scale * input.abs() + 1.0) ** 2 #torch.abs(input)
        return grad, None

spike_fn = SuSpike.apply




class BaseLayer(nn.Module):
    def __init__(self, neuron_params, device):
        super(BaseLayer, self).__init__()
        self.device = device
        self.dtype = torch.float32
        self.bias = False
        
        self.alpha = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=False)
        self.beta = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=False)
        self.th = nn.Parameter(torch.empty((1), device=self.device, dtype=self.dtype), requires_grad=False)
        
        nn.init.constant_(self.alpha, np.exp(-1 / neuron_params['tau_s']))
        nn.init.constant_(self.beta, np.exp(-1 / neuron_params['tau_m']))
        nn.init.constant_(self.th, neuron_params['theta'])

    
    def initialise_state(self, input_spikes):
        return
    
    def update_psp(self, input_spikes, output_spikes=None):
        return
    
        
    def state_update(self, input_spikes):
        # shape of input_spikes B x C x H x W or B x C x N
        out_spike = spike_fn(self.mem, self.th)
        rst = out_spike.detach().clone().bool()
        self.mem = (self.beta * self.mem * (1-rst.type(self.dtype))+ (1 - self.beta)*self.syn)
        self.syn = self.alpha * self.syn + self.update_psp(input_spikes, out_spike)
        
        return out_spike



class SpikingConvLayer(BaseLayer):
    def __init__(self, neuron_params, input_size, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device='cuda:0'):
        super().__init__(neuron_params, device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.input_size = input_size #(H,W)
        output_height = int(np.floor((input_size[0] - kernel_size + 2*padding) / stride) + 1)
        output_width = int(np.floor((input_size[1] - kernel_size + 2*padding) / stride) + 1)
        self.output_size = (output_height, output_width)
        
        self.num_neurons = self.output_size[0] * self.output_size[1] * out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data)
    
    def initialise_state(self, B):
        self.syn = torch.zeros(B, self.out_channels, self.output_size[0], self.output_size[1], device=self.device, dtype=self.dtype)
        self.mem = torch.zeros(B, self.out_channels, self.output_size[0], self.output_size[1], device=self.device, dtype=self.dtype)

    def update_psp(self, input_spikes):
        return self.conv(input_spikes)
    
    def state_update(self, input_spikes):
        # shape of input_spikes B x C x H x W
        out_spike = spike_fn(self.mem, self.th)
        rst = out_spike.detach().clone().bool()
        self.mem = (self.beta * self.mem * (1-rst.type(self.dtype))+ (1 - self.beta)*self.syn)
        self.syn = self.alpha * self.syn + self.conv(input_spikes)
            
        return out_spike

      
class SpikingPoolingLayer(BaseLayer):
    def __init__(self, neuron_params, input_size, channels, kernel_size=2, device='cuda:0'):
        super().__init__(neuron_params, device)
        #input_size #(H,W)
        self.kernel_size = kernel_size
        self.output_size = (int(input_size[0]//kernel_size), int(input_size[1]//kernel_size))
        self.input_size = input_size
        self.channels = channels
        self.num_neurons = self.output_size[0] * self.output_size[1] * self.channels 
        
        
    def initialise_state(self, B): 
        self.syn = torch.zeros(B, self.channels, self.output_size[0], self.output_size[1], device=self.device, dtype=self.dtype)
        self.mem = torch.zeros(B, self.channels, self.output_size[0], self.output_size[1], device=self.device, dtype=self.dtype)

        
    def update_psp(self, input_spikes):
        return F.avg_pool2d(input_spikes, 2)
    
    def state_update(self, input_spikes):
        # shape of input_spikes B x C x H x W
        out_spike = spike_fn(self.mem, self.th)
        
        rst = out_spike.detach().clone().bool()
        self.mem = (self.beta * self.mem * (1-rst.type(self.dtype))+ (1 - self.beta)*self.syn)
        self.syn = self.alpha * self.syn + F.avg_pool2d(input_spikes, 2) #self.update_psp(input_spikes, out_spike)
            
        return out_spike



class SpikingLayer(BaseLayer):
    def __init__(self, neuron_params, input_size, output_size,  device, param_trainable=False):
        super().__init__(neuron_params, device)
        self.output_size = output_size
        self.num_neurons = output_size
        
        if param_trainable:
            self.alpha = nn.Parameter(torch.empty((1, self.output_size), device=self.device, dtype=self.dtype), requires_grad=True)
            self.beta = nn.Parameter(torch.empty((1, self.output_size), device=self.device, dtype=self.dtype), requires_grad=True)

            nn.init.constant_(self.alpha, np.exp(-1 / neuron_params['tau_s']))
            nn.init.constant_(self.beta, np.exp(-1 / neuron_params['tau_m']))
        # Linear weights
        self.w = nn.Parameter(
            torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))
        nn.init.xavier_uniform_(self.w)

    
    def initialise_state(self, B):
        self.syn = torch.zeros((B, self.output_size), device=self.device, dtype=self.dtype)
        self.mem = torch.zeros((B, self.output_size), device=self.device, dtype=self.dtype)
        

    def update_psp(self, input_spikes):
        return torch.matmul(input_spikes, self.w)

    def state_update(self, input_spikes):
        # shape of input_spikes B x N   
        out_spike = spike_fn(self.mem, self.th)
        
        rst = out_spike.detach().clone().bool()

        self.mem = (self.beta * self.mem * (1-rst.type(self.dtype))+ (1 - self.beta)*self.syn)
        self.syn = self.alpha * self.syn + torch.matmul(input_spikes, self.w)

        return out_spike



class RecurrentSpikingLayer(BaseLayer):
    def __init__(self, neuron_params, input_size, output_size, device, param_trainable=False):
        super().__init__(neuron_params, device)
        self.output_size = output_size
        self.num_neurons = output_size
        
        if param_trainable:
            self.alpha = nn.Parameter(torch.empty((1, self.output_size), device=self.device, dtype=self.dtype), requires_grad=True)
            self.beta = nn.Parameter(torch.empty((1, self.output_size), device=self.device, dtype=self.dtype), requires_grad=True)

            nn.init.constant_(self.alpha, np.exp(-1 / neuron_params['tau_s']))
            nn.init.constant_(self.beta, np.exp(-1 / neuron_params['tau_m']))

        # Create variables
        self.w = nn.Parameter(
            torch.empty((input_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))
        self.v = nn.Parameter(
            torch.empty((output_size, output_size), device=self.device, dtype=self.dtype, requires_grad=True))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.v)
            
    def initialise_state(self, B):
        
        self.syn = torch.zeros((B, self.output_size), device=self.device, dtype=self.dtype)
        self.mem = torch.zeros((B, self.output_size), device=self.device, dtype=self.dtype)

    def update_psp(self, input_spikes, out_spike):
        return torch.matmul(input_spikes, self.w) + torch.mm(out_spike, self.v)


    def state_update(self, input_spikes):
        # shape of input_spikes B x N   
        out_spike = spike_fn(self.mem, self.th)
        
        rst = out_spike.detach().clone().bool()
        
        self.mem = (self.beta * self.mem * (1-rst.type(self.dtype))+ (1 - self.beta)*self.syn)
        self.syn = self.alpha * self.syn + torch.matmul(input_spikes, self.w) + torch.mm(out_spike, self.v)

        return out_spike

