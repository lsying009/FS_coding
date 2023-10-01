import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

class Time2FSTFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output_times, t_inf):
        # B x N xT
        first_times, indices = torch.min(output_times, dim=-1)
        
        ctx.t_inf = t_inf
        ctx.save_for_backward(output_times, indices)
        return first_times
    @staticmethod
    def backward(ctx, propagated_time_error):
        
        # B x N --> B x N x T
        output_times, indices = ctx.saved_tensors

        #-------------- assign to the all the neuron for t_inf -------------#
        dead_mask =(output_times >= ctx.t_inf).all(-1, keepdim=False)
        time_gradient = torch.zeros_like(output_times)
        all_time_gradient = propagated_time_error.unsqueeze(-1).repeat(1,1,output_times.size()[-1])
        time_gradient = time_gradient.scatter_(2, indices.unsqueeze(-1), propagated_time_error.unsqueeze(-1))
        time_gradient[dead_mask] = all_time_gradient[dead_mask]

        
        return time_gradient, None, None

t2first = Time2FSTFunc.apply


class Spike2TimeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output_spikes, output_potentials, D=16, A=200, device='cuda:0'):
        B, N, T = output_spikes.shape
        time_steps = torch.arange(1, T+1, device=device)

        # the neuron with the max membrane potential has potential to fire the first (dead neurons)
        max_u, _ = output_potentials.max(-1)
        _, indices_neuron = torch.sort(max_u, dim=-1, descending=True)

        M_neuron = torch.arange(1., N+1., device=device).repeat(B,1)
        X_neuron = torch.zeros_like(M_neuron).scatter_(1, indices_neuron, M_neuron)
        X_neuron = X_neuron.unsqueeze(-1).repeat(1,1,T)
        
        # the time step with higher potential has potential to fire the first for dead neurons
        _, indices_time = torch.sort(output_potentials, dim=-1, descending=True)
        M_time = torch.arange(0, 0.01*T,step=0.01, device=device).unsqueeze(0).repeat(B,N,1)
        X_time = X_neuron.scatter_add_(2, indices_time, M_time)
        
        output_times =  time_steps * output_spikes \
                        + (T+X_time) * (1-output_spikes)
        
        # define Gaussian kernel for error assignment
        sigma = T//D
        length = min(sigma*6+1, 2*(T//2)+1)
        kernel = signal.gaussian(length, std=sigma) 
        # Nx1xW
        kernel = torch.from_numpy(kernel).float().reshape(1,1,-1).repeat(output_times.shape[1],1,1).to(device)
        
        ctx.kernel = kernel
        ctx.A = A

        return output_times
        
    def backward(ctx, propagated_time_error):
        grad = -ctx.A * propagated_time_error

        # ctx.kernel
        # propagated_time_error B x N x T,  N x 1 x W --> B x N x T
        spike_gradient = F.conv1d(grad, ctx.kernel, padding='same', groups=ctx.kernel.shape[0])

        return spike_gradient, None, None, None, None



class Spike2Time(nn.Module):
    def __init__(self, loss_params, theta=1.0, device='cuda:0'):
        super(Spike2Time, self).__init__()
        self.device = device
        self.D = loss_params['FS']['D']
        self.A = loss_params['FS']['A']
        self.theta = theta
  
    
    def forward(self, output_spikes, output_potentials=None, theta=None):
        output_times = Spike2TimeFunc.apply(output_spikes, output_potentials, self.D, self.A, self.device)
        first_times = Time2FSTFunc.apply(output_times, (output_spikes.shape[-1]+1))
        return first_times
 