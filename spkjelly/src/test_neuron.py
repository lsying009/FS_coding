## codes come from https://github.com/Gausx/DVS_Gestures.git
# from binascii import a2b_base64
import torch
import torch.nn as nn
import numpy as np

from spikingjelly.activation_based import cuda_utils, functional, surrogate, neuron
from neuron_ex import CuLIFNode, ParametricCuLIFNode, CuAdLIFNode, AdLIFNode
from surrogate_ex import MySoftSign

rand_seed = 3
#------------------- Test precision -----------------#
@torch.no_grad()
def max_error(x: torch.Tensor, y: torch.Tensor):
    # error, indices = torch.max((x - y).abs())
    error_mean = (x - y).abs().mean()
    error_max = (x - y).abs().max()
    indices = (x - y).abs().argmax()
    return error_mean, error_max, x[(x-y).abs()>0.01], y[(x-y).abs()>0.01]


T = 8
N = 64
C = 32 * 32 * 32
device = 'cuda:0'
x_seq = torch.rand([T, N, C], device=device, requires_grad=True)


# net_torch = neuron.LIFNode(tau= 10., v_threshold=1.,  detach_reset=True, backend='torch', step_mode='m')
# net_torch = ParametricCuLIFNode(tau_mem= 5., tau_syn = 5., v_threshold=1., surrogate_function=MySoftSign(),\
#     detach_reset=True, backend='torch', step_mode='m')
# y_torch = net_torch(x_seq)
# y_torch.sum().backward()
# x_grad_torch = x_seq.grad.clone()
# x_seq.grad.zero_()
torch.manual_seed(rand_seed)
net_torch = AdLIFNode(tau_mem= 5., v_threshold=0.5, output_size= C, surrogate_function=MySoftSign(),\
    detach_reset=True, backend='torch', step_mode='m',  decay_mode='s', store_v_seq=True).to(device)
# net_torch = AdLIFNode(tau_mem= 5., tau_syn = 5., v_threshold=0.5, surrogate_function=MySoftSign(), \
#     detach_reset=True, backend='torch', step_mode='m', store_v_seq=True)

# y_torch = []
# for i in range(T):
#     y = net_torch(x_seq[i])
#     y_torch.append(y)
# y_torch = torch.stack(y_torch, 0)

y_torch = net_torch(x_seq)
y_torch.sum().backward()
x_grad_torch = x_seq.grad.clone()
# grad_h_seq = torch.autograd.grad(loss, h_seq, allow_unused=True)[0]
x_seq.grad.zero_()

torch.manual_seed(rand_seed)
# net_cupy = CuAdLIFNode(tau_mem= 5., tau_syn = 5., v_threshold=0.5, output_size= C, surrogate_function=MySoftSign(), \
#     detach_reset=True, backend='cupy', step_mode='m', decay_mode='s', store_v_seq=True)
net_cupy = AdLIFNode(tau_mem= 5., v_threshold=0.5, output_size= C, surrogate_function=MySoftSign(), \
    detach_reset=True, backend='cupy', step_mode='m', decay_mode='s', store_v_seq=True)

# net_cupy = CuLIFNode(tau_mem= 5., tau_syn = 5., v_threshold=0.5, surrogate_function=MySoftSign(), \
#     detach_reset=True, backend='cupy', step_mode='m', store_v_seq=True)



y_cupy = net_cupy(x_seq)
y_cupy.sum().backward()
x_grad_cupy = x_seq.grad.clone()
x_seq.grad.zero_()


print(y_cupy[:,0,0:5])
print(y_torch[:,0,0:5])
print(x_grad_cupy[:,0,0:5])
print(x_grad_torch[:,0,0:5])
print('max error of y_seq', max_error(y_cupy, y_torch))
print('max error of x_seq.grad', max_error(x_grad_cupy, x_grad_torch))
print('max error of h_seq.grad', max_error(net_cupy.h_seq, net_torch.h_seq))
print('max error of v_seq.grad', max_error(net_cupy.v_seq, net_torch.v_seq))



#------------------- Test speed -----------------#
def forward_backward(net: torch.nn.Module, x_seq: torch.Tensor):
    y_seq = net(x_seq)
    y_seq.sum().backward(retain_graph=True)
    x_seq.grad.zero_()
    functional.reset_net(net)



# repeats = 16
# net_cupy = CuAdLIFNode(tau_mem= 5., tau_syn = 5., v_threshold=1., surrogate_function=MySoftSign(),  detach_reset=True, backend='cupy', step_mode='m')
# net_torch = CuAdLIFNode(tau_mem= 5., tau_syn = 5., v_threshold=1., surrogate_function=MySoftSign(), detach_reset=True, backend='torch', step_mode='m')
# net_torch.to(device)

# for dtype in [torch.float, torch.half]:
#     for T in [2, 4, 8, 16, 32]:
#         x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=dtype)

#         t_torch = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_torch, x_seq)
#         t_cupy = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_cupy, x_seq)

#         print(f'dtype={dtype}, T={T},'.ljust(30), f't_torch / t_cupy = {round(t_torch / t_cupy, 2)}')

#         # functional.reset_net(net_torch)
#         # functional.reset_net(net_cupy)


