import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from spikingjelly.activation_based import functional
from spikingjelly.activation_based import base, layer
from .neuron_ex import CuLIFNode, ParametricCuLIFNode, CuAdLIFNode, AdLIFNode
from .surrogate_ex import MySoftSign
from .container_ex import AddLinearRecurrentContainer
from .time_encoding import Spike2Time

def weight_init_xavier_uniform(m):
    for name, param in m.named_parameters():
        if ('weight' in name) or ('bias' in name):
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)


class SSC_Attention(nn.Module, base.StepModule):
    def __init__(self, n_channel: int):

        super().__init__()
        self.step_mode = 'm'    # used in activation_based SpikingJelly

        # if self.dimension == 4:
        #     self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.n_channels = n_channel
        self.recv_C = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_channel, n_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')
        x_seq_C = x_seq.transpose(0, 1) # x_seq_C.shape = [B, T, N] or [B, T, C, H, W]

        if x_seq.dim() == 3:
            recv_h_C = self.recv_C(x_seq_C)
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)
            
            x_seq = x_seq * D
            
        elif x_seq.dim() == 5:
            avgout_C = F.adaptive_avg_pool2d(x_seq_C, 1).view([x_seq_C.shape[0], x_seq_C.shape[1], x_seq_C.shape[2]]) # avgout_C.shape = [N, T, C]
            recv_h_C = self.recv_C(avgout_C.transpose(1, 2))
            D_ = 1 - self.sigmoid(recv_h_C)
            D = D_.transpose(0, 1)
            
            x_seq = x_seq * D[:, :, :, None, None]
            
        return x_seq


class CLA(nn.Module):
    def __init__(self, kernel_size: int = 2, T: int = 8, device='cuda:0'):
        super().__init__()
        # self.conv = nn.Conv1d(in_channels=T, out_channels=T,
        #                       kernel_size=kernel_size, padding='same', bias=False)
        self.conv1d_weight = nn.Parameter(torch.empty((1, 1, kernel_size), device=device))
        nn.init.xavier_uniform_(self.conv1d_weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, B, N] or [T, B, C, H, W], but got input with shape {x_seq.shape}')
        T = x_seq.shape[0]
        x_seq_C = x_seq.transpose(0, 1) # x_seq_C.shape = [B, T, N] or [B, T, C, H, W]
        if x_seq.dim() == 3:
            # conv_t_out = self.conv(x_seq_C).permute(1, 0, 2)
            conv_t_out = F.conv1d(x_seq_C, self.conv1d_weight.repeat(T,1,1), padding='same', groups=T).permute(1, 0, 2)
            
            out = self.sigmoid(conv_t_out)
            x_seq = x_seq * out
            
        if x_seq.dim() == 5:
            x = torch.mean(x_seq_C, dim=[3, 4])
            conv_t_out = self.conv(x).permute(1, 0, 2)
            out = self.sigmoid(conv_t_out)
            x_seq = x_seq * out[:, :, :, None, None]
            
        return x_seq


def order_spikes(spikes):
    # input T,B,C
    spikes_shape = spikes.shape
    T = spikes_shape[0]
    
    zero_mask = (spikes ==0)

    # indices = spikes.nonzero(as_tuple=True)
    # weight_matrix = torch.zeros_like(spikes)
    # weight_matrix[indices[0], indices[1], indices[2]] = indices[0].float()+1
    # weight_matrix[zero_mask] = T+1
    # min_values, _ = torch.min(weight_matrix, dim=0, keepdim=True)
    # weight_matrix = (T - weight_matrix + min_values)/T
    # weight_matrix[zero_mask] = 1
    # ordered_spikes = spikes * weight_matrix

    weights = torch.arange(1,T+1).flip(0).to(spikes)/T
    ordered_spikes = torch.einsum('a,ab->ab', weights, spikes.view(T,-1)).reshape(spikes_shape)
    # weight_matrix = weights.unsqueeze(-1).unsqueeze(-1).repeat(1, spikes_shape[1], spikes_shape[2])
    # weight_matrix[zero_mask] = 1
    # ordered_spikes = spikes * weight_matrix
    
    return ordered_spikes


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
            recurrent=False, attention=False, BN=False, dropout=None, 
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
                    store_v_seq = store_v_seq, store_I_seq = store_I_seq)
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
    if attention:
        spikefc.append(SSC_Attention(input_size))
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


class SHD_FCNet(nn.Module):
    def __init__(self,input_size, out_size, hidden_size, \
                neuron_params1, neuron_params2, loss_params, device='cuda:0', \
                is_spike_train=False, is_spike_count=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc_layers = nn.ModuleList([])
        self.num_neurons = []
   
        for i, h_size in enumerate(self.hidden_size):
            self.fc_layers.append(
                SpikeFC(neuron_params1, 'cuadlif', input_size, h_size,
                        decay_mode='m', attention=False, BN=True, dropout=0.25)
                        # store_v_seq=True, store_I_seq=True)
                )
            self.num_neurons.append(h_size)
            input_size = h_size

        self.fc_layers.append(
            SpikeFC(neuron_params2, 'culif', input_size, out_size, BN=False, store_v_seq=True) #, store_I_seq=True)
            # SpikeFC(neuron_params2, 'pculif', input_size, out_size, decay_mode='m', store_v_seq=True)
        )
        self.num_neurons.append(out_size)
        
        weight_init_xavier_uniform(self.fc_layers)
        
        # self.spike2time = Spike2Time(loss_params=loss_params,device=device)
        loss_params['TTFS']['eta'] = 5
        loss_params['TTFS']['constant'] = 1
        self.spike2time = Spike2Time(loss_params=loss_params,theta=neuron_params2['theta'], 
                                     device=device)
        
        self.init_D = loss_params['TTFS']['D']
        self.max_D = 64
        self.final_epoch = 80
        
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=device)
        self.num_neurons = np.array(self.num_neurons)
        self.device = device
        self.is_spike_train = is_spike_train
        self.is_spike_count = is_spike_count
    
    def decrease_sig(self):
        print('D', self.spike2time.D)
        if self.spike2time.D <= self.max_D:
            alpha = (self.max_D/self.init_D)**(1/self.final_epoch)
            self.spike2time.D *= alpha
        
    def forward(self, x: torch.Tensor):
        
        # B x C x T --> T x B x C
        x = x.permute(2, 0, 1)
        
        # if self.is_spike_count:
        B = x.shape[1] # batch size
        self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=self.device)
        for i, fc in enumerate(self.fc_layers):
            # x = order_spikes(x)
            x = fc(x)
            self.spike_counts[i] = x.sum()/B
            # if i < 2:
            #     plot_one_neuron_v_I_s(fc[-2].v_seq[:,0,175].cpu(), fc[-2].I_seq[:,0,175].cpu(), x[:,0,175].cpu(), v_threshold=1, v_reset=0,
            #                         dpi=200)
            #     plt.show()
        # plot_one_neuron_v_I_s(fc[-1].v_seq[:,0,0].cpu(), fc[-1].I_seq[:,0,0].cpu(), x[:,0,0].cpu(), v_threshold=10, v_reset=0,
        #                         dpi=200)
        # plt.show()

        # output_times = self.spike2time(x.permute(1, 2, 0), self.fc_layers[-1].sub_module[-1].h_seq.permute(1, 2, 0)) ####### TimedelayContainer
        output_times = self.spike2time(x.permute(1, 2, 0), self.fc_layers[-1][-1].h_seq.permute(1, 2, 0)) #, self.fc_layers[-1][-1].v_threshold) ####### sequential only
        # firing_rate = self.fc_layers[-1][-1].h_seq.mean(0) #softmax(-1)
        firing_rate = x.mean(0)
        # firing_rate = order_spikes(x).mean(0)

        outputs = [output_times, firing_rate]
        
        
        if self.is_spike_train:
            outputs.append([x.permute(1, 2, 0)])
            
        if self.is_spike_count:
            avg_spk_cnt = self.spike_counts.sum() / self.num_neurons.sum()
            outputs.append(avg_spk_cnt)
            
        return outputs

        #--------- for voting layer -----------#
        # output_times = F.avg_pool1d(output_times.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)
        # firing_rate = F.avg_pool1d(firing_rate.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)



class FCNet(nn.Module):
    def __init__(self,input_size, out_size, hidden_size, \
                neuron_params1, neuron_params2, loss_params, device='cuda:0', \
                dropout=None, recurrent=False, is_spike_train=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc_layers = nn.ModuleList([])
        self.num_neurons = []
   
        for h_size in self.hidden_size:
            self.fc_layers.append(
                SpikeFC(neuron_params1, 'cuadlif', input_size, h_size, 
                        decay_mode='m', attention=False, BN=True, dropout=dropout,
                        recurrent=recurrent, backend='cupy'
                        )
                )
            self.num_neurons.append(h_size)
            input_size = h_size

        self.fc_layers.append( #BN=False
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
            outputs.append([x.permute(1, 2, 0)])
            
        return outputs


# class SHD_FCNet(nn.Module):
#     def __init__(self,input_size, out_size, hidden_size, \
#                 neuron_params1, neuron_params2, loss_params, device='cuda:0', \
#                 is_spike_train=False, is_spike_count=False):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.fc_layers = nn.ModuleList([])
#         self.num_neurons = []
   
#         for i, h_size in enumerate(self.hidden_size):
#             self.fc_layers.append(
#                 # nn.Sequential(
#                 # AddLinearRecurrentContainer(                
#                 nn.Sequential(
#                     # SSC_Attention(input_size),
#                     layer.SeqToANNContainer(
#                     nn.Linear(input_size, h_size, bias=False),
#                     nn.BatchNorm1d(h_size)),
#                     CuAdLIFNode(tau_mem= neuron_params1['tau_m'], tau_syn = neuron_params1['tau_s'], tau_osc = 100.0, v_threshold=neuron_params1['theta'],  \
#                     surrogate_function=MySoftSign(), \
#                     output_size=h_size, detach_reset=True, backend='cupy', step_mode='m', decay_mode='m'),
#                     # ParametricCuLIFNode(tau_mem= neuron_params1['tau_m'], tau_syn = neuron_params1['tau_s'], v_threshold=neuron_params1['theta'],  \
#                     # surrogate_function=MySoftSign(), \
#                     # output_size=h_size, detach_reset=True, backend='cupy', step_mode='m', decay_mode='m', 
#                     # ),
#                     layer.Dropout(0.25, step_mode='m')
#                 )
#                 # h_size, bias=False, step_mode='m'
#                 # )
#             )
#             self.num_neurons.append(h_size)
#             input_size = h_size

#         self.fc_layers.append(
#             nn.Sequential(
#                     layer.SeqToANNContainer(
#                     nn.Linear(h_size, out_size, bias=False),
#                     nn.BatchNorm1d(out_size)),
#                     # layer.Linear(h_size, out_size, bias=False, step_mode='m'),
#                     CuLIFNode(tau_mem= neuron_params2['tau_m'], tau_syn = neuron_params2['tau_s'], v_threshold=neuron_params2['theta'],  \
#                     surrogate_function=MySoftSign(), \
#                     detach_reset=True, backend='cupy', step_mode='m', store_v_seq=True),
#                 )
#         )
#         self.num_neurons.append(out_size)
        
#         weight_init_xavier_uniform(self.fc_layers)
        
#         # self.spike2time = Spike2Time(loss_params=loss_params,device=device)

#         self.spike2time = Spike2Time(loss_params=loss_params,theta=neuron_params2['theta'], 
#                                      device=device)
#         self.init_D = loss_params['TTFS']['D']
#         self.max_D = 125
#         self.final_epoch = 200
        
        
#         self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=device)
#         self.num_neurons = np.array(self.num_neurons)
#         self.device = device
#         self.is_spike_train = is_spike_train
#         self.is_spike_count = is_spike_count
    
#     def decrease_sig(self):
#         print('D', self.spike2time.D)
#         if self.spike2time.D <= self.max_D:
#             alpha = (self.max_D/self.init_D)**(1/self.final_epoch)
#             self.spike2time.D *= alpha

        
#     def forward(self, x: torch.Tensor):
        
#         # B x C x T --> T x B x C
#         x = x.permute(2, 0, 1)
        
#         # if self.is_spike_count:
#         B = x.shape[1] # batch size
#         self.spike_counts = torch.zeros(len(self.fc_layers), dtype=torch.float32, device=self.device)
#         for i, fc in enumerate(self.fc_layers):
#             x = fc(x)
#             self.spike_counts[i] = x.sum()/B


#         output_times = self.spike2time(x.permute(1, 2, 0), self.fc_layers[-1][-1].h_seq.permute(1, 2, 0)) ####### sequential only
#         firing_rate = x.mean(0)

#         outputs = [output_times, firing_rate]
        
        
#         if self.is_spike_train:
#             outputs.append([x.permute(1, 2, 0)])
            
#         if self.is_spike_count:
#             avg_spk_cnt = self.spike_counts.sum() / self.num_neurons.sum()
#             outputs.append(avg_spk_cnt)
            
#         return outputs

 