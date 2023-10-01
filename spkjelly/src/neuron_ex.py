import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import surrogate, neuron
from typing import Callable
from .neuron_kernel_ex import *


class CuLIFNode(neuron.BaseNode):
    def __init__(self, tau_mem: float = 5., tau_syn: float = 5., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, store_I_seq: bool = False):
        """
        The Current-based Leaky Integrate-and-Fire (CuLIF) neuron
        """
        assert isinstance(tau_mem, float) and tau_mem > 0. and isinstance(tau_syn, float) and tau_syn > 0.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.decay_mem = math.exp(-1./tau_mem)
        self.decay_syn = math.exp(-1./tau_syn)
        
        # self.I
        self.register_memory('I', 0.)
        self.register_memory('h', 0.)
        self.store_I_seq = store_I_seq

    
    def I_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.I, float):
            I_init = self.I
            self.I = torch.full_like(x.data, I_init) # input_size, fill_value

    def h_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.I, float):
            h_init = self.h
            self.h = torch.full_like(x.data, h_init) # input_size, fill_value


    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        # add extra variables
        return super().extra_repr() + f', tau_mem={self.tau_mem}' + f', tau_syn={self.tau_syn}'+ f', dycay_mem={self.decay_mem}' + f', decay_syn={self.decay_syn}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None or self.v_reset == 0.:
            self.v, self.I = self.neuronal_charge_decay_input_reset0(x, self.v, self.I, self.decay_mem, self.decay_syn)
        else:
            self.v, self.I = self.neuronal_charge_decay_input(x, self.v, self.I, self.v_reset, self.decay_mem, self.decay_syn)
        self.h = self.v.clone().detach()
        
    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, I: torch.Tensor, decay_mem: float, decay_syn: float):
        v = I + (v - I) * decay_mem
        I = decay_syn * I + x 
        return v, I

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_reset: float, decay_mem: float, decay_syn: float):
        v = I + (v - v_reset - I) * decay_mem + v_reset
        I = decay_syn * I + x
        
        return v, I

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                            v_reset: float, decay_mem: float, decay_syn: float):
        
        h = I + (v - v_reset - I) * decay_mem + v_reset
        I = decay_syn * I + x
        spike = (h >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * h
        return spike, v, I, h


    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                            decay_mem: float, decay_syn: float):
        
        h = I + (v - I) * decay_mem
        I = decay_syn * I + x
        spike = (h >= v_threshold).to(x)
        v = h - spike * v_threshold
        return spike, v, I, h

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                           v_reset: float, decay_mem: float, decay_syn: float):
        spike_seq = torch.zeros_like(x_seq)

        for t in range(x_seq.shape[0]):
            v = I + (v - v_reset - I) * decay_mem + v_reset
            I = decay_syn * I + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v, I

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                           v_reset: float, decay_mem: float, decay_syn: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        h_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = I + (v - v_reset - I) * decay_mem + v_reset
            I = decay_syn * I + x_seq[t]
            h_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, I, v_seq, h_seq
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_I_seq(x_seq: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                           v_reset: float, decay_mem: float, decay_syn: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        h_seq = torch.zeros_like(x_seq)
        I_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = I + (v - v_reset - I) * decay_mem + v_reset
            I = decay_syn * I + x_seq[t]
            h_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
            I_seq[t] = I
        return spike_seq, v, I, v_seq, h_seq, I_seq


    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                        decay_mem: float, decay_syn: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = I + (v - I) * decay_mem
            I = decay_syn * I + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v, I

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                        decay_mem: float, decay_syn: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        h_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = I + (v - I) * decay_mem
            I = decay_syn * I + x_seq[t]
            h_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, I, v_seq, h_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_I_seq(x_seq: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_threshold: float,
                                                        decay_mem: float, decay_syn: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        h_seq = torch.zeros_like(x_seq)
        I_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = I + (v - I) * decay_mem
            I = decay_syn * I + x_seq[t]
            h_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
            I_seq[t] = I
        return spike_seq, v, I, v_seq, h_seq, I_seq


    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            self.I_float_to_tensor(x)
            self.h_float_to_tensor(x)
            self.v_float_to_tensor(x)
            self.neuronal_charge(x)
            self.h = self.v.clone().detach()
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            return spike
        else:
            self.v_float_to_tensor(x)
            self.I_float_to_tensor(x)
            self.h_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v, self.I, self.h = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v, self.I,
                                                                                            self.v_threshold, self.decay_mem, self.decay_syn)
            else:
                spike, self.v, self.I, self.h = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v, self.I,
                                                                                            self.v_threshold,
                                                                                            self.v_reset, self.decay_mem, self.decay_syn)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        
        if self.training:
            if self.backend == 'torch':
                # return super().multi_step_forward(x_seq)
                ## modified with store_I_seq
                T = x_seq.shape[0]
                y_seq = []
                if self.store_v_seq:
                    v_seq = []
                    h_seq = []
                if self.store_I_seq:
                    I_seq = []
                for t in range(T):
                    y = self.single_step_forward(x_seq[t])
                    y_seq.append(y)
                    if self.store_v_seq:
                        v_seq.append(self.v)
                        h_seq.append(self.h)
                    if self.store_I_seq:
                        I_seq.append(self.I)

                if self.store_v_seq:
                    self.v_seq = torch.stack(v_seq)
                    self.h_seq = torch.stack(h_seq)
                if self.store_I_seq:
                    self.I_seq = torch.stack(I_seq)

                return torch.stack(y_seq)
            
            elif self.backend == 'cupy':

                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype):
                    self.forward_kernel = CuLIFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype):
                    self.backward_kernel = CuLIFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x_seq[0])
                self.I_float_to_tensor(x_seq[0])

                # x_seq, v_init, I_init, v_th, v_reset, mem_decay, syn_decay, forward_kernel, backward_kernel
                spike_seq, v_seq, I_seq, h_seq = CuLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0), self.I.flatten(0),
                                                                     self.v_threshold, self.v_reset, self.decay_mem, self.decay_syn, ######
                                                                     self.forward_kernel,
                                                                     self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)
                I_seq = I_seq.reshape(x_seq.shape)
                h_seq = h_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq
                    self.h_seq = h_seq
                if self.store_I_seq:
                    self.I_seq = I_seq

                self.v = v_seq[-1].clone()
                self.I = I_seq[-1].clone()

                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            self.I_float_to_tensor(x_seq[0])

            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.I, self.v_seq, self.h_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                        x_seq, self.v, self.I, self.v_threshold, self.decay_mem, self.decay_syn)
                elif self.store_I_seq:
                    spike_seq, self.v, self.I, self.v_seq, self.h_seq, self.I_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_I_seq(
                        x_seq, self.v,  self.I, self.v_threshold, self.decay_mem, self.decay_syn)
                else:
                    spike_seq, self.v, self.I = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.v, self.I,
                                                                                                self.v_threshold,
                                                                                                self.decay_mem, self.decay_syn)
            else:
                if self.store_I_seq:
                    spike_seq, self.v, self.I, self.v_seq, self.h_seq, self.I_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_I_seq(
                        x_seq, self.v, self.I, self.v_threshold, self.v_reset, self.decay_mem, self.decay_syn)
                elif self.store_v_seq:
                    spike_seq, self.v, self.I, self.v_seq, self.h_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
                        x_seq, self.v, self.I, self.v_threshold, self.v_reset, self.decay_mem, self.decay_syn)
                else:
                    spike_seq, self.v, self.I = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, self.v, self.I, self.v_threshold,  self.v_reset,
                                                                                         self.decay_mem, self.decay_syn)
            return spike_seq



class ParametricCuLIFNode(neuron.BaseNode):
    def __init__(self, tau_mem: float = 5., tau_syn: float = 5., v_threshold: float = 1.,
                 v_reset: float = 0., output_size: tuple or list or int = 1, surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', decay_mode='s', backend='torch', store_v_seq: bool = False, store_I_seq: bool = False):
        """
        The Parametric Current-based Leaky Integrate-and-Fire (PCuLIF) neuron, the learnable parameters are implemented by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:
        
        where :math:`exp(-\\frac{1}{\\tau}) = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.
        decay_mode: 'm' -- multiple learned decay params for each output neuron
                    's' -- only one single learned value for decay
        """

        assert isinstance(tau_mem, float) and tau_mem > 1.
        assert isinstance(tau_syn, float) and tau_syn > 1.
        
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        
        self.decay_mode = decay_mode
        self.output_size = output_size
        # ensure init decay exp(-1/tau)
        init_decay_mem = -math.log(math.exp(1./tau_mem) - 1.) # math.exp(-1./tau_mem)
        init_decay_syn = -math.log(math.exp(1./tau_syn) - 1.) #math.exp(-1./tau_syn)
        
        if decay_mode == 'm':
            if isinstance(output_size, int):
                self.decay_mem = nn.Parameter(torch.empty((1, output_size)), requires_grad=True)
                self.decay_syn = nn.Parameter(torch.empty((1, output_size)), requires_grad=True)
            else: #tuple or list
                self.decay_mem = nn.Parameter(torch.empty((1,)+ output_size), requires_grad=True)
                self.decay_syn = nn.Parameter(torch.empty((1,)+ output_size), requires_grad=True)
            nn.init.constant_(self.decay_mem, init_decay_mem)
            nn.init.constant_(self.decay_syn, init_decay_syn)
        elif decay_mode == 's':
            self.decay_mem = nn.Parameter(torch.as_tensor(init_decay_mem))
            self.decay_syn = nn.Parameter(torch.as_tensor(init_decay_syn))
        else:
            assert decay_mode == 's' or decay_mode == 'm'
            
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_syn = -1. / torch.log(self.decay_syn.sigmoid()).detach()

        # self.I
        self.register_memory('I', 0.)
        self.register_memory('h', 0.)
        self.store_I_seq = store_I_seq

        
    
    def I_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.I, float):
            I_init = self.I
            self.I = torch.full_like(x.data, I_init) # input_size, fill_value
    
    def h_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.h, float):
            h_init = self.h
            self.h = torch.full_like(x.data, h_init) # input_size, fill_value

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau_mem={self.tau_mem}' + f', tau_syn={self.tau_syn}'+ f', decay_mem={self.decay_mem}' + f', decay_syn={self.decay_syn}'
    
    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None or self.v_reset == 0.:
            self.v, self.I = self.neuronal_charge_decay_input_reset0(x, self.v, self.I, self.decay_mem, self.decay_syn)
        else:
            self.v, self.I = self.neuronal_charge_decay_input(x, self.v, self.I, self.v_reset, self.decay_mem, self.decay_syn)
        self.h = self.v.clone()
        
    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, I: torch.Tensor, decay_mem: torch.Tensor, decay_syn: torch.Tensor):
        v = I + (v - I) * decay_mem.sigmoid()
        I = decay_syn.sigmoid() * I + x 
        return v, I

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, I: torch.Tensor, v_reset: float, decay_mem: torch.Tensor, decay_syn: torch.Tensor):
        v = I + (v - v_reset - I) * decay_mem.sigmoid() + v_reset
        I = decay_syn.sigmoid() * I + x
        
        return v, I

        
    def single_step_forward(self, x: torch.Tensor):
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_syn = -1. / torch.log(self.decay_syn.sigmoid()).detach()

        self.I_float_to_tensor(x)
        self.h_float_to_tensor(x)
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        self.saved_input = x_seq
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_syn = -1. / torch.log(self.decay_syn.sigmoid()).detach()
        
        

        if self.backend == 'torch':
            ## modified with store_I_seq
            T = x_seq.shape[0]
            y_seq = []
            if self.store_v_seq:
                v_seq = []
                h_seq = []
            if self.store_I_seq:
                I_seq = []
            
            
            for t in range(T):
                y = self.single_step_forward(x_seq[t])
                y_seq.append(y)
                if self.store_v_seq:
                    v_seq.append(self.v)
                    h_seq.append(self.h)
                if self.store_I_seq:
                    I_seq.append(self.I)
            
            
            if self.store_v_seq:
                self.v_seq = torch.stack(v_seq)
                self.h_seq = torch.stack(h_seq)

            if self.store_I_seq:
                self.I_seq = torch.stack(I_seq)

            
            return torch.stack(y_seq)
        
        elif self.backend == 'cupy':

            hard_reset = self.v_reset is not None
            if x_seq.dtype == torch.float:
                dtype = 'float'
            elif x_seq.dtype == torch.half:
                dtype = 'half2'
            else:
                raise NotImplementedError(x_seq.dtype)
            
            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype):
                self.forward_kernel = ParametricCuLIFNodeFPTTKernel(hard_reset=hard_reset, decay_mode=self.decay_mode, dtype=dtype)

            if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype):
                self.backward_kernel = ParametricCuLIFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, decay_mode=self.decay_mode, 
                                                                     hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)


            self.v_float_to_tensor(x_seq[0])
            self.I_float_to_tensor(x_seq[0])

            # x_seq, v_init, I_init, v_th, v_reset, mem_decay, syn_decay, forward_kernel, backward_kernel
            spike_seq, v_seq, I_seq, h_seq = ParametricCuLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0), self.I.flatten(0),
                                                                    self.v_threshold, self.v_reset, self.decay_mem.sigmoid().flatten(0).to(x_seq), self.decay_syn.sigmoid().flatten(0).to(x_seq), ######
                                                                    self.decay_syn.numel(), # batch_size
                                                                    self.decay_mode, 
                                                                    self.forward_kernel,
                                                                    self.backward_kernel)
            
            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            I_seq = I_seq.reshape(x_seq.shape)
            h_seq = h_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq
                self.h_seq = h_seq
            if self.store_I_seq:
                self.I_seq = I_seq

            self.v = v_seq[-1].clone()
            self.I = I_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)

  
class CuAdLIFNode(neuron.BaseNode):
    def __init__(self, tau_mem: float = 5., tau_syn: float = 5., tau_osc: float = 100., v_threshold: float = 1.,
                 v_reset: float = 0., output_size: tuple or list or int = 1, surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', decay_mode='m', backend='torch', store_v_seq: bool = False, store_I_seq: bool = False):
        """
        The Current-based Adaptive Leaky Integrate-and-Fire (CuAdLIF) neuron
        decay_mode: 'm' -- multiple learned decay params for each output neuron
                    's' -- only one single learned value for decay
        """

        assert isinstance(tau_mem, float) and tau_mem > 0.
        assert isinstance(tau_syn, float) and tau_syn > 0.
        assert isinstance(tau_osc, float) and tau_osc > 0.
         
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        
        self.decay_mode = decay_mode
        self.output_size = output_size
        # ensure init decay exp(-1/tau)
        init_decay_mem = -math.log(math.exp(1./tau_mem) - 1.) # math.exp(-1./tau_mem)
        init_decay_syn = -math.log(math.exp(1./tau_syn) - 1.) # math.exp(-1./tau_syn)
        init_decay_osc = -math.log(math.exp(1./tau_osc) - 1.)
        self.a_lim = [0,1]
        self.b_lim = [0,1] #[0,2]
    
    
        if decay_mode == 'm':
            param_size = (1, output_size) if isinstance(output_size, int) else (1,)+ output_size
            self.decay_mem = nn.Parameter(torch.empty(param_size), requires_grad=True)
            self.decay_syn = nn.Parameter(torch.empty(param_size), requires_grad=True)
            self.decay_osc = nn.Parameter(torch.empty(param_size), requires_grad=True)
            self.a = nn.Parameter(torch.empty(param_size), requires_grad=True)
            self.b = nn.Parameter(torch.empty(param_size), requires_grad=True)

            nn.init.constant_(self.decay_mem, init_decay_mem)
            nn.init.constant_(self.decay_syn, init_decay_syn)
            nn.init.constant_(self.decay_osc, init_decay_osc)
            nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
            nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
            
        elif decay_mode == 's':
            self.decay_mem = nn.Parameter(torch.as_tensor(init_decay_mem))
            self.decay_syn = nn.Parameter(torch.as_tensor(init_decay_syn))
            self.decay_osc = nn.Parameter(torch.as_tensor(init_decay_osc))
            self.a = nn.Parameter(torch.Tensor(1).uniform_(self.a_lim[0], self.a_lim[1]))
            self.b = nn.Parameter(torch.Tensor(1).uniform_(self.b_lim[0], self.b_lim[1]))
        else:
            assert decay_mode == 's' or decay_mode == 'm'
            
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_syn = -1. / torch.log(self.decay_syn.sigmoid()).detach()
        self.tau_osc = -1. / torch.log(self.decay_osc.sigmoid()).detach()

        # self.I
        self.register_memory('I', 0.)
        self.register_memory('w', 0.)
        self.register_memory('h', 0.)
        self.register_memory('s', 0.)
        self.store_I_seq = store_I_seq

        
    
    def I_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.I, float):
            I_init = self.I
            self.I = torch.full_like(x.data, I_init) # input_size, fill_value

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            w_init = self.w
            self.w = torch.full_like(x.data, w_init) # input_size, fill_value
    
    def h_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.h, float):
            h_init = self.h
            self.h = torch.full_like(x.data, h_init) # input_size, fill_value

    def s_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.s, float):
            s_init = self.s
            self.s = torch.full_like(x.data, s_init) # input_size, fill_value
    
    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau_mem={self.tau_mem}' + f', tau_syn={self.tau_syn}'+ f', tau_osc={self.tau_osc}'+ f', decay_mem={self.decay_mem}' + f', decay_syn={self.decay_syn}'+ f', decay_osc={self.decay_osc}'+ f', a={self.a}'+ f', b={self.b}'
    
    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None or self.v_reset == 0.:
            self.v, self.I, self.w = self.neuronal_charge_decay_input_reset0(x, self.s, self.v, self.h, self.I, self.w, self.decay_mem, self.decay_syn, self.decay_osc, self.a, self.b)
        else:
            self.v, self.I, self.w = self.neuronal_charge_decay_input(x, self.s, self.v, self.h, self.I, self.w, self.v_reset, self.decay_mem, self.decay_syn, self.decay_osc, self.a, self.b)
        self.h = self.v.clone()
        
    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, s: torch.Tensor, v: torch.Tensor, h: torch.Tensor, I: torch.Tensor,  w: torch.Tensor, 
                                           decay_mem: torch.Tensor, decay_syn: torch.Tensor, decay_osc: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        # v_pre = v.clone()
        v = I - w + (v - I + w) * decay_mem.sigmoid()
        I = decay_syn.sigmoid() * I + x
        w = decay_osc.sigmoid() * w + a * h + b * s
        return v, I, w

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, s: torch.Tensor, v: torch.Tensor, h: torch.Tensor, I: torch.Tensor, w: torch.Tensor, v_reset: float, decay_mem: torch.Tensor, decay_syn: torch.Tensor, decay_osc: torch.Tensor, a: torch.Tensor,  b: torch.Tensor):
        # v_pre = v.clone()
        v = I - w + (v - v_reset - I + w) * decay_mem.sigmoid() + v_reset
        I = decay_syn.sigmoid() * I + x
        w = decay_osc.sigmoid() * w + a * h  + b * s
        
        return v, I, w


        
    def single_step_forward(self, x: torch.Tensor):
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_syn = -1. / torch.log(self.decay_syn.sigmoid()).detach()
        self.tau_osc = -1. / torch.log(self.decay_osc.sigmoid()).detach()
        
        self.a.data.clamp_(min=self.a_lim[0], max=self.a_lim[1])
        self.b.data.clamp_(min=self.b_lim[0], max=self.b_lim[1])
        
        self.I_float_to_tensor(x)
        self.w_float_to_tensor(x)
        self.s_float_to_tensor(x)
        self.h_float_to_tensor(x)
        ################
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        self.s = self.neuronal_fire()
        self.neuronal_reset(self.s)
        return self.s

    def multi_step_forward(self, x_seq: torch.Tensor):
        self.saved_input = x_seq
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_syn = -1. / torch.log(self.decay_syn.sigmoid()).detach()
        self.tau_osc = -1. / torch.log(self.decay_osc.sigmoid()).detach()
       
        if self.backend == 'torch':
            ## modified with store_I_seq
            T = x_seq.shape[0]
            y_seq = []
            if self.store_v_seq:
                v_seq = []
                h_seq = []
            if self.store_I_seq:
                I_seq = []

            for t in range(T):
                y = self.single_step_forward(x_seq[t])
                y_seq.append(y)
                
                if self.store_v_seq:
                    v_seq.append(self.v)
                    h_seq.append(self.h)
                if self.store_I_seq:
                    I_seq.append(self.I)

            if self.store_v_seq:
                self.v_seq = torch.stack(v_seq)
                self.h_seq = torch.stack(h_seq)

            if self.store_I_seq:
                self.I_seq = torch.stack(I_seq)
            
            return torch.stack(y_seq)
        
        elif self.backend == 'cupy':

            hard_reset = self.v_reset is not None
            if x_seq.dtype == torch.float:
                dtype = 'float'
            elif x_seq.dtype == torch.half:
                dtype = 'half2'
            else:
                raise NotImplementedError(x_seq.dtype)
            
            self.a.data.clamp_(min=self.a_lim[0], max=self.a_lim[1])
            self.b.data.clamp_(min=self.b_lim[0], max=self.b_lim[1])
            
            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype):
                self.forward_kernel = CuAdLIFNodeFPTTKernel(hard_reset=hard_reset, decay_mode=self.decay_mode, dtype=dtype)

            if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype):
                self.backward_kernel = CuAdLIFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, decay_mode=self.decay_mode, 
                                                                     hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)


            self.v_float_to_tensor(x_seq[0])
            self.I_float_to_tensor(x_seq[0])
            self.w_float_to_tensor(x_seq[0])

            # x_seq, v_init, I_init, v_th, v_reset, mem_decay, syn_decay, forward_kernel, backward_kernel
            spike_seq, v_seq, I_seq, h_seq = CuAdLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0), self.I.flatten(0), self.w.flatten(0),
                                                                    self.v_threshold, self.v_reset, 
                                                                    self.decay_mem.sigmoid().flatten(0).to(x_seq), self.decay_syn.sigmoid().flatten(0).to(x_seq), ######
                                                                    self.decay_osc.sigmoid().flatten(0).to(x_seq), self.a.flatten(0).to(x_seq), self.b.flatten(0).to(x_seq), ######
                                                                    self.decay_syn.numel(), # output_size
                                                                    self.decay_mode, 
                                                                    self.forward_kernel,
                                                                    self.backward_kernel)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            I_seq = I_seq.reshape(x_seq.shape)
            h_seq = h_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq
                self.h_seq = h_seq
            if self.store_I_seq:
                self.I_seq = I_seq

            self.v = v_seq[-1].clone()
            self.I = I_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class AdLIFNode(neuron.BaseNode):
    def __init__(self, tau_mem: float = 5., tau_osc: float = 100., v_threshold: float = 1.,
                 v_reset: float = 0., output_size: tuple or list or int = 1, surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', decay_mode='m', backend='torch', store_v_seq: bool = False):
        """
        The Adaptive Leaky Integrate-and-Fire (AdLIF) neuron, proposed by Bittar, A., and Garner, P. N. (2022). A surrogate gradient spiking baseline for speech
        command recognition
        
        decay_mode: 'm' -- multiple learned decay params for each output neuron
                    's' -- only one single learned value for decay
        """

        assert isinstance(tau_mem, float) and tau_mem > 0.
        assert isinstance(tau_osc, float) and tau_osc > 0.
        
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        
        self.decay_mode = decay_mode
        self.output_size = output_size
        # ensure init decay exp(-1/tau)
        init_decay_mem = -math.log(math.exp(1./tau_mem) - 1.) # math.exp(-1./tau_mem)
        init_decay_osc = -math.log(math.exp(1./tau_osc) - 1.)
        self.a_lim = [0,1]
        self.b_lim = [0,1] #[0,2]
    
    
        if decay_mode == 'm':
            param_size = (1, output_size) if isinstance(output_size, int) else (1,)+ output_size
            self.decay_mem = nn.Parameter(torch.empty(param_size), requires_grad=True)
            self.a = nn.Parameter(torch.empty(param_size), requires_grad=True)
            self.b = nn.Parameter(torch.empty(param_size), requires_grad=True)

            nn.init.constant_(self.decay_mem, init_decay_mem)
            nn.init.constant_(self.decay_osc, init_decay_osc)
            nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
            nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
            
        elif decay_mode == 's':
            self.decay_mem = nn.Parameter(torch.as_tensor(init_decay_mem))
            self.decay_osc = nn.Parameter(torch.as_tensor(init_decay_osc))
            self.a = nn.Parameter(torch.Tensor(1).uniform_(self.a_lim[0], self.a_lim[1]))
            self.b = nn.Parameter(torch.Tensor(1).uniform_(self.b_lim[0], self.b_lim[1]))
        else:
            assert decay_mode == 's' or decay_mode == 'm'
            
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_osc = -1. / torch.log(self.decay_osc.sigmoid()).detach()

        self.register_memory('w', 0.)
        self.register_memory('h', 0.)
        self.register_memory('s', 0.)

        

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            w_init = self.w
            self.w = torch.full_like(x.data, w_init) # input_size, fill_value
    
    def h_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.h, float):
            h_init = self.h
            self.h = torch.full_like(x.data, h_init) # input_size, fill_value

    def s_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.s, float):
            s_init = self.s
            self.s = torch.full_like(x.data, s_init) # input_size, fill_value
    
    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau_mem={self.tau_mem}' + f', decay_osc={self.decay_osc}'+ f', a={self.a}'+ f', b={self.b}'
    
    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None or self.v_reset == 0.:
            self.v, self.w = self.neuronal_charge_decay_input_reset0(x, self.s, self.v, self.h, self.w, self.decay_mem, self.decay_osc, self.a, self.b)
        else:
            self.v, self.w = self.neuronal_charge_decay_input(x, self.s, self.v, self.h, self.w, self.v_reset, self.decay_mem, self.decay_osc, self.a, self.b)
        self.h = self.v.clone()
        
    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, s: torch.Tensor, v: torch.Tensor, h: torch.Tensor,  w: torch.Tensor, 
                                           decay_mem: torch.Tensor, decay_osc: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        # v_pre = v.clone()
        v = x - w + (v - x + w) * decay_mem.sigmoid()
        w = decay_osc.sigmoid() * w + a * h + b * s
        return v, w

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, s: torch.Tensor, v: torch.Tensor, h: torch.Tensor, w: torch.Tensor, v_reset: float, decay_mem: torch.Tensor, decay_osc: torch.Tensor, a: torch.Tensor,  b: torch.Tensor):
        # v_pre = v.clone()
        v = x - w + (v - v_reset - x + w) * decay_mem.sigmoid() + v_reset
        w = decay_osc.sigmoid() * w + a * h  + b * s
        
        return v, w

        
    def single_step_forward(self, x: torch.Tensor):
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_osc = -1. / torch.log(self.decay_osc.sigmoid()).detach()
        
        self.a.data.clamp_(min=self.a_lim[0], max=self.a_lim[1])
        self.b.data.clamp_(min=self.b_lim[0], max=self.b_lim[1])
        
        self.w_float_to_tensor(x)
        self.s_float_to_tensor(x)
        self.h_float_to_tensor(x)
        ################
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        self.s = self.neuronal_fire()
        self.neuronal_reset(self.s)
        return self.s

    def multi_step_forward(self, x_seq: torch.Tensor):
        self.saved_input = x_seq
        self.tau_mem = -1. / torch.log(self.decay_mem.sigmoid()).detach()
        self.tau_osc = -1. / torch.log(self.decay_osc.sigmoid()).detach()
       
        if self.backend == 'torch':
            T = x_seq.shape[0]
            y_seq = []
            if self.store_v_seq:
                v_seq = []
                h_seq = []

            for t in range(T):
                y = self.single_step_forward(x_seq[t])
                y_seq.append(y)
                
                if self.store_v_seq:
                    v_seq.append(self.v)
                    h_seq.append(self.h)

            if self.store_v_seq:
                self.v_seq = torch.stack(v_seq)
                self.h_seq = torch.stack(h_seq)


            return torch.stack(y_seq)
        
        elif self.backend == 'cupy':

            hard_reset = self.v_reset is not None
            if x_seq.dtype == torch.float:
                dtype = 'float'
            elif x_seq.dtype == torch.half:
                dtype = 'half2'
            else:
                raise NotImplementedError(x_seq.dtype)
            
            self.a.data.clamp_(min=self.a_lim[0], max=self.a_lim[1])
            self.b.data.clamp_(min=self.b_lim[0], max=self.b_lim[1])
            
            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype):
                self.forward_kernel = AdLIFNodeFPTTKernel(hard_reset=hard_reset, decay_mode=self.decay_mode, dtype=dtype)

            if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype):
                self.backward_kernel = AdLIFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, decay_mode=self.decay_mode, 
                                                                     hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)


            self.v_float_to_tensor(x_seq[0])
            self.w_float_to_tensor(x_seq[0])

            # x_seq, v_init, I_init, v_th, v_reset, mem_decay, syn_decay, forward_kernel, backward_kernel
            spike_seq, v_seq, h_seq = AdLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0), self.I.flatten(0), self.w.flatten(0),
                                                                    self.v_threshold, self.v_reset, 
                                                                    self.decay_mem.sigmoid().flatten(0).to(x_seq),
                                                                    self.decay_osc.sigmoid().flatten(0).to(x_seq), self.a.flatten(0).to(x_seq), self.b.flatten(0).to(x_seq), ######
                                                                    self.decay_mode, 
                                                                    self.forward_kernel,
                                                                    self.backward_kernel)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            h_seq = h_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq
                self.h_seq = h_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)
