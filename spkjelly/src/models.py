import torch
import torch.nn as nn
import numpy as np


from spikingjelly.activation_based import layer
from .neuron_ex import CuLIFNode, ParametricCuLIFNode, CuAdLIFNode, AdLIFNode
from .surrogate_ex import MySoftSign
from .container_ex import AddLinearRecurrentContainer
from .time_encoding import Spike2Time



def weight_init_xavier_uniform(m):
    for name, param in m.named_parameters():
        if ('weight' in name) or ('bias' in name):
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)


def SpikeConv(neuron_params, neuron_type, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, BN=False, backend='cupy', step_mode='m'):
    if neuron_type == 'pculif':
        neuron = ParametricCuLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], v_threshold=neuron_params['theta'],  \
                    surrogate_function=MySoftSign(), \
                    detach_reset=True, backend=backend, step_mode=step_mode, decay_mode='s', 
                    )
    else: #'culif'
        neuron = CuLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], v_threshold=neuron_params['theta'],  \
                 surrogate_function=MySoftSign(), \
                detach_reset=True, backend=backend, step_mode=step_mode) 

    if BN:
        spikeconv = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
            ),
            neuron
        )
    else:
        spikeconv = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, step_mode='m'),
            neuron
        )
    return spikeconv


def SpikePool(neuron_params, neuron_type, kernel_size=2, backend='cupy', step_mode='m'):
    if neuron_type == 'pculif':
        neuron = ParametricCuLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], v_threshold=neuron_params['theta'],  \
                    surrogate_function=MySoftSign(), \
                    detach_reset=True, backend=backend, step_mode=step_mode, decay_mode='s', 
                    )
    else: #'culif'
        neuron = CuLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], v_threshold=neuron_params['theta'],  \
                 surrogate_function=MySoftSign(), \
                detach_reset=True, backend=backend, step_mode=step_mode) 
    spikepool = nn.Sequential(
        layer.MaxPool2d(kernel_size=kernel_size, step_mode=step_mode),
        neuron
    )
    return spikepool


def SpikeFC(neuron_params, neuron_type, input_size, out_size, bias=False, 
            recurrent=False, BN=False, dropout=None, 
            backend='cupy', step_mode='m', decay_mode='m', store_v_seq=False, store_I_seq=False):
    if recurrent:
        backend = 'torch'
        neuron_step_mode = 's'
    else:
        neuron_step_mode = step_mode
    if neuron_type == 'adlif':
        neuron = AdLIFNode(tau_mem= neuron_params['tau_m'], tau_osc = 100.0, v_threshold=neuron_params['theta'],  \
                    surrogate_function=MySoftSign(), \
                    output_size=out_size, detach_reset=True, backend=backend, step_mode=neuron_step_mode, decay_mode=decay_mode, 
                    store_v_seq = store_v_seq)
    elif neuron_type == 'cuadlif':
        neuron = CuAdLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], tau_osc = 100.0, v_threshold=neuron_params['theta'],  \
                    surrogate_function=MySoftSign(), \
                    output_size=out_size, detach_reset=True, backend=backend, step_mode=neuron_step_mode, decay_mode=decay_mode, 
                    store_v_seq = store_v_seq, store_I_seq = store_I_seq) 
    elif neuron_type == 'pculif':
        neuron = ParametricCuLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], v_threshold=neuron_params['theta'],  \
                    surrogate_function=MySoftSign(), \
                    output_size=out_size, detach_reset=True, backend=backend, step_mode=neuron_step_mode, decay_mode=decay_mode, 
                    store_v_seq = store_v_seq, store_I_seq = store_I_seq) 
    else: #'culif'
        neuron = CuLIFNode(tau_mem= neuron_params['tau_m'], tau_syn = neuron_params['tau_s'], v_threshold=neuron_params['theta'],  \
                 surrogate_function=MySoftSign(), \
                detach_reset=True, backend=backend, step_mode=neuron_step_mode, 
                store_v_seq=store_v_seq, store_I_seq = store_I_seq) 
    
    spikefc = []  
    if BN:
        spikefc.append(layer.SeqToANNContainer(
                nn.Linear(input_size, out_size, bias=bias),
                nn.BatchNorm1d(out_size)))
    else:
        spikefc.append(layer.Linear(input_size, out_size, bias=bias, step_mode=neuron_step_mode))
    
    spikefc.append(neuron)
    
    if dropout is not None:
        spikefc.append(layer.Dropout(dropout, step_mode=step_mode)) #, step_mode=step_mode
        

    if recurrent:
        spikefc = AddLinearRecurrentContainer(
                nn.Sequential(*spikefc),
                out_size, bias=bias, step_mode=step_mode
                )
    else:
        spikefc =  nn.Sequential(*spikefc)
    
    return spikefc


class DVSGestureNet(nn.Module):
    def __init__(self,input_size, out_size, \
                neuron_params1, neuron_params2, loss_params, device='cuda:0', \
                is_spike_train=False):
        super().__init__()
        self.input_size = (input_size[1], input_size[2]) #[2,32,32]
        self.out_size = out_size

        neuron_type = 'culif'
        self.conv_layers = nn.ModuleList([
            SpikeConv(neuron_params1, neuron_type, input_size[0], 64, BN=False),
            SpikeConv(neuron_params1, neuron_type, 64, 128),
            SpikePool(neuron_params1, neuron_type, 2),
            SpikeConv(neuron_params1, neuron_type, 128, 128),
            SpikePool(neuron_params1, neuron_type, 2)
            ]
        )
        
        self.fc_layers = nn.ModuleList([
            SpikeFC(neuron_params2, neuron_type, 128 * int(self.input_size[0]/4) * int(self.input_size[1]/4), 128, recurrent=True),
            SpikeFC(neuron_params2, 'culif', 128, out_size, store_v_seq=True),
        ])
        
 
        self.num_neurons=[
            64 * self.input_size[0] * self.input_size[1],
            128 * self.input_size[0] * self.input_size[1],
            128 * int(self.input_size[0]/2) * int(self.input_size[1]/2),
            128 * int(self.input_size[0]/2) * int(self.input_size[1]/2),
            128 * int(self.input_size[0]/4) * int(self.input_size[1]/4),
            128,
            self.out_size
        ]

        weight_init_xavier_uniform(self.conv_layers)
        weight_init_xavier_uniform(self.fc_layers)
        
        self.spike2time = Spike2Time(loss_params=loss_params,device=device)
        
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=device)
        self.num_neurons = np.array(self.num_neurons)
        self.device = device
        self.is_spike_train = is_spike_train
        
    def forward(self, x: torch.Tensor):
        # B x 2 x H x W x T --> T x B x 2 x H x W 
        x = x.permute(4, 0, 1, 2, 3)
        
        B = x.shape[1] # batch size
        self.spike_counts = torch.zeros(len(self.num_neurons), dtype=torch.float32, device=self.device)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            self.spike_counts[i] = x.sum()/B
        # T x B x C x H x W --> T x B x N
        x = x.flatten(start_dim=2, end_dim=-1)
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            self.spike_counts[len(self.conv_layers) + i] = x.sum()/B

        output_times = self.spike2time(x.permute(1, 2, 0), self.fc_layers[-1][-1].h_seq.permute(1, 2, 0))
        firing_rate = x.mean(0)

        outputs = [output_times, firing_rate]
        
        if self.is_spike_train:
            outputs.append(x.permute(1, 2, 0))
            
        return outputs


class DVSPlaneNet(nn.Module):
    def __init__(self,input_size, out_size, \
                neuron_params1, neuron_params2, loss_params, device='cuda:0', \
                is_spike_train=False):
        super().__init__()
        self.input_size = (input_size[1], input_size[2]) #[2,32,32]
        self.out_size = out_size

        neuron_type = 'culif'
        self.conv_layers = nn.ModuleList([
            SpikeConv(neuron_params1, neuron_type, input_size[0], 32, kernel_size=5, stride=2, padding=2),
            SpikeConv(neuron_params1, neuron_type, 32, 64),
            SpikePool(neuron_params1, neuron_type, 2),
            SpikeConv(neuron_params1, neuron_type, 64, 128),
            SpikePool(neuron_params1, neuron_type, 2)
        ])
        
        self.fc_layers = nn.ModuleList([
            SpikeFC(neuron_params2, neuron_type, 128 * int(self.input_size[0]//8) * int(self.input_size[1]//8), 256, recurrent=True),
            SpikeFC(neuron_params2, 'culif', 256, out_size, store_v_seq=True),
        ])
    
        self.num_neurons=[
            32 * int(self.input_size[0]/2) * int(self.input_size[1]/2),
            64 * int(self.input_size[0]/2) * int(self.input_size[1]/2),
            64 * int(self.input_size[0]/4) * int(self.input_size[1]/4),
            128 * int(self.input_size[0]/4) * int(self.input_size[1]/4),
            128 * int(self.input_size[0]//8) * int(self.input_size[1]//8),
            256,
            self.out_size
        ]

        weight_init_xavier_uniform(self.conv_layers)
        weight_init_xavier_uniform(self.fc_layers)
        
        self.spike2time = Spike2Time(loss_params=loss_params,device=device)
        
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=device)
        self.num_neurons = np.array(self.num_neurons)
        self.device = device
        self.is_spike_train = is_spike_train
        
    def forward(self, x: torch.Tensor):
        # B x 2 x H x W x T --> T x B x 2 x H x W 
        x = x.permute(4, 0, 1, 2, 3)
        
        B = x.shape[1] # batch size
        self.spike_counts = torch.zeros(len(self.num_neurons), dtype=torch.float32, device=self.device)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            self.spike_counts[i] = x.sum()/B
        # T x B x C x H x W --> T x B x N
        x = x.flatten(start_dim=2, end_dim=-1)
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            self.spike_counts[len(self.conv_layers) + i] = x.sum()/B

        output_times = self.spike2time(x.permute(1, 2, 0), self.fc_layers[-1][-1].h_seq.permute(1, 2, 0))
        firing_rate = x.mean(0)

        outputs = [output_times, firing_rate]
        
        if self.is_spike_train:
            outputs.append(x.permute(1, 2, 0))
             
        return outputs


class FCNet(nn.Module):
    """
    Implementation of FCNet with AdLIF neurons
    """
    def __init__(self,input_size, out_size, hidden_size, \
                neuron_params1, neuron_params2, loss_params, device='cuda:0', \
                is_spike_train=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc_layers = nn.ModuleList([])
        self.num_neurons = []
   
        for h_size in self.hidden_size:
            self.fc_layers.append(
                SpikeFC(neuron_params1, 'adlif', input_size, h_size, 
                        decay_mode='m', BN=True, dropout=0.25,
                        recurrent=False, backend='cupy'
                        )
                )
            self.num_neurons.append(h_size)
            input_size = h_size

        self.fc_layers.append(
            SpikeFC(neuron_params2, 'culif', input_size, out_size, BN=False, store_v_seq=True, backend='cupy')
        )
        self.num_neurons.append(out_size)
        weight_init_xavier_uniform(self.fc_layers)
        
        self.spike2time = Spike2Time(loss_params=loss_params,device=device)
        
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=device)
        self.num_neurons = np.array(self.num_neurons)
        self.device = device
        self.is_spike_train = is_spike_train
        
    def forward(self, x: torch.Tensor):
        
        # B x C x T --> T x B x C
        x = x.permute(2, 0, 1)
        
        B = x.shape[1]
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=self.device)
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            self.spike_counts[i] = x.sum()/B

        output_times = self.spike2time(x.permute(1, 2, 0), self.fc_layers[-1][-1].h_seq.permute(1, 2, 0)) ####### sequential only
        firing_rate = x.mean(0)

        outputs = [output_times, firing_rate]
        
        if self.is_spike_train:
            outputs.append(x.permute(1, 2, 0))
            
        return outputs
