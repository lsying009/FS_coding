import torch
import torch.nn as nn
import numpy as np


from .layers import SpikingConvLayer, SpikingPoolingLayer, \
    RecurrentSpikingLayer, SpikingLayer
from .time_encoding import Spike2Time


class SpikeDVSGestureCNN(nn.Module):
    def __init__(self, input_size, out_size, neuron_params, neuron_fc_params, loss_params, device, test=False):
        super(SpikeDVSGestureCNN, self).__init__()
        self.input_size = (input_size[1], input_size[2]) #[2,32,32]
        self.out_size = out_size

        self.conv1 = SpikingConvLayer(neuron_params, self.input_size, input_size[0], 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.conv2 = SpikingConvLayer(neuron_params, self.input_size, 64, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.pool1 = SpikingPoolingLayer(neuron_params, self.input_size, 128,  kernel_size=2, device=device)
        self.conv3 = SpikingConvLayer(neuron_params, (self.input_size[0]//2, self.input_size[1]//2), 128, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.pool2 = SpikingPoolingLayer(neuron_params, (self.input_size[0]//2, self.input_size[1]//2), 128, kernel_size=2, device=device)

        self.fc1 = RecurrentSpikingLayer(neuron_fc_params, 128 * int(self.input_size[0]/4) * int(self.input_size[1]/4), 128, device)
        self.fc2 = SpikingLayer(neuron_fc_params, 128, self.out_size, device)
        
        self.spike2time = Spike2Time(loss_params=loss_params, device=device)
        

        self.device = device
        self.test = test

        self.spike_counts = torch.zeros(7, dtype=torch.float32, device=device)
        self.num_neurons = np.array([self.conv1.num_neurons, self.conv2.num_neurons, self.pool1.num_neurons, self.conv3.num_neurons,self.pool2.num_neurons,self.fc1.num_neurons,self.fc2.num_neurons])
    
    def forward(self, input):
        # temp variable define / initial state should be zero
        step = input.size()[-1]
        B = input.size()[0]
        self.conv1.initialise_state(B)
        self.conv2.initialise_state(B)
        self.pool1.initialise_state(B)
        self.conv3.initialise_state(B)
        self.pool2.initialise_state(B)
        self.fc1.initialise_state(B)
        self.fc2.initialise_state(B)
        
        self.spike_counts = torch.zeros(7, dtype=torch.float32, device=self.device)
        
        output_spikes = torch.zeros(B, self.out_size, step, device=self.device)
        output_potentials = torch.zeros(B, self.out_size, step, device=self.device)

        for t in range(step):
            in_t = input[..., t]
            conv1_s = self.conv1.state_update(in_t)
            conv2_s = self.conv2.state_update(conv1_s)
            pool1_s = self.pool1.state_update(conv2_s)
            conv3_s = self.conv3.state_update(pool1_s)
            pool2_s = self.pool2.state_update(conv3_s)
            
            fc_in_spikes = pool2_s.view(pool2_s.shape[0], -1)
            fc1_s = self.fc1.state_update(fc_in_spikes)
            fc2_s = self.fc2.state_update(fc1_s)
            
            output_spikes[...,t] = fc2_s
            if t < step-1:
                output_potentials[...,t+1] = self.fc2.mem
            
            # if t < 120:
            self.spike_counts[0] += conv1_s.sum()/B
            self.spike_counts[1] += conv2_s.sum()/B
            self.spike_counts[2] += pool1_s.sum()/B
            self.spike_counts[3] += conv3_s.sum()/B
            self.spike_counts[4] += pool2_s.sum()/B
            self.spike_counts[5] += fc1_s.sum()/B
            self.spike_counts[6] += fc2_s.sum()/B
        

        output_times = self.spike2time(output_spikes, output_potentials)
        firing_rate = output_spikes.mean(-1)

        if self.test: 
            return output_times, firing_rate, output_spikes
        else:
            return output_times, firing_rate 


class SpikeDVSPlaneCNN(nn.Module):
    def __init__(self, input_size, out_size, neuron_params, neuron_fc_params, loss_params, device, test=False):
        super(SpikeDVSPlaneCNN, self).__init__()
        self.input_size = (input_size[1], input_size[2]) #[2,32,32]
        self.out_size = out_size
        
        base_ch = 32
        self.conv1 = SpikingConvLayer(neuron_params, self.input_size,  input_size[0], base_ch, kernel_size=5, stride=2, padding=2, bias=False, device=device) #30x38
        self.conv2 = SpikingConvLayer(neuron_params, self.conv1.output_size, base_ch, 2*base_ch, kernel_size=3, stride=1, padding=1, bias=False, device=device) #30x38
        self.pool1 = SpikingPoolingLayer(neuron_params, self.conv1.output_size, 2*base_ch, kernel_size=2, device=device) #15x19
        self.conv3 = SpikingConvLayer(neuron_params, self.pool1.output_size, 2*base_ch, 4*base_ch, kernel_size=3, stride=1, padding=1, bias=False, device=device) # 15x19
        self.pool2 = SpikingPoolingLayer(neuron_params, self.conv3.output_size, 4*base_ch, kernel_size=2, device=device) #7x9 / 8x10?

        self.fc1_size = 256
        self.fc1 = RecurrentSpikingLayer(neuron_fc_params,4*base_ch*self.pool2.output_size[0] * self.pool2.output_size[1], self.fc1_size, device)
        self.fc2 = SpikingLayer(neuron_fc_params, self.fc1_size, self.out_size, device)
        
        self.spike2time = Spike2Time(loss_params=loss_params,  device=device)
        

        self.device = device
        self.test = test

        self.spike_counts = torch.zeros(7, dtype=torch.float32, device=device)
        self.num_neurons = np.array([self.conv1.num_neurons, self.conv2.num_neurons,  self.pool1.num_neurons, self.conv3.num_neurons, self.pool2.num_neurons,self.fc1.num_neurons,self.fc2.num_neurons])
    
    def forward(self, input):
        # temp variable define / initial state should be zero
        step = input.size()[-1]
        B = input.size()[0]
        # if self.test:
        self.conv1.initialise_state(B)
        self.conv2.initialise_state(B)
        self.pool1.initialise_state(B)
        self.conv3.initialise_state(B)
        self.pool2.initialise_state(B)
        self.fc1.initialise_state(B)
        self.fc2.initialise_state(B)
        
        self.spike_counts = torch.zeros(len(self.spike_counts), dtype=torch.float32, device=self.device)
        
        output_spikes = torch.zeros(B, self.out_size, step, device=self.device)
        output_potentials = torch.zeros(B, self.out_size, step, device=self.device)

        for t in range(step):
            in_t = input[..., t]
            conv1_s = self.conv1.state_update(in_t) 
            conv2_s = self.conv2.state_update(conv1_s)
            pool1_s = self.pool1.state_update(conv2_s)
            conv3_s = self.conv3.state_update(pool1_s)
            pool2_s = self.pool2.state_update(conv3_s)
            
            fc_in_spikes = pool2_s.view(pool2_s.shape[0], -1)
            fc1_s = self.fc1.state_update(fc_in_spikes)
            fc2_s = self.fc2.state_update(fc1_s)
            
            output_spikes[...,t] = fc2_s
            if t < step-1:
                output_potentials[...,t+1] = self.fc2.mem
            
            # if t < 120:
            self.spike_counts[0] += conv1_s.sum()/B
            self.spike_counts[1] += conv2_s.sum()/B
            self.spike_counts[2] += pool1_s.sum()/B
            self.spike_counts[3] += conv3_s.sum()/B
            self.spike_counts[4] += pool2_s.sum()/B
            self.spike_counts[5] += fc1_s.sum()/B
            self.spike_counts[6] += fc2_s.sum()/B

        output_times = self.spike2time(output_spikes, output_potentials) #self.fc2.spk_rec, self.fc2.mem_rec
        
        firing_rate = output_spikes.mean(-1)

        if self.test: 
            return output_times, firing_rate, output_spikes
        else:
            return output_times, firing_rate 


class SpikeFCNet(nn.Module):
    def __init__(self, input_size, out_size, hidden_size, neuron_params1, neuron_params2, loss_params, device, test=False):
        super(SpikeFCNet, self).__init__()
        self.out_size = out_size

        self.hidden_size = hidden_size
        self.input_size = input_size if isinstance(input_size, int) else input_size[0]*input_size[1]*input_size[2]
        
        self.fc_layers = nn.ModuleList([])
        self.num_neurons = []
        input_size = self.input_size
        for h_size in self.hidden_size:
            self.fc_layers.append(
                RecurrentSpikingLayer(neuron_params1, input_size, h_size, device, param_trainable=True)
                )
            self.num_neurons.append(h_size)
            input_size = h_size
        self.fc_layers.append(
            SpikingLayer(neuron_params2, h_size, self.out_size, device, param_trainable=False)
                              )
        self.num_neurons.append(self.out_size)
        self.spike2time = Spike2Time(loss_params=loss_params,device=device)
        
        self.device = device
        self.test = test
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=device)
        self.num_neurons = np.array(self.num_neurons)

    def forward(self, input):
        # temp variable define / initial state should be zero
        step = input.size()[-1]
        B = input.size()[0]
        
        for layer in self.fc_layers:
            layer.initialise_state(B)
        
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=self.device)
        
        output_spikes = torch.zeros(B, self.out_size, step, device=self.device)
        output_potentials = torch.zeros(B, self.out_size, step, device=self.device)

        in_t = input.view(input.shape[0], -1, input.shape[-1])
        for t in range(step):
            fc_s = in_t[...,t]
            for i, layer in enumerate(self.fc_layers):
                fc_s = layer.state_update(fc_s)
                self.spike_counts[i] += fc_s.sum()/B
            
            output_spikes[...,t] = fc_s
            if t<step-1:
                output_potentials[...,t+1] = self.fc_layers[-1].mem
            
        output_times = self.spike2time(output_spikes, output_potentials)
        firing_rate = output_spikes.mean(-1)
        
        if self.test:
            return output_times, firing_rate, output_spikes
        else:
            return  output_times, firing_rate

