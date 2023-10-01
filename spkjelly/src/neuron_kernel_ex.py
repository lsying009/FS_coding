import torch
from typing import Callable


from spikingjelly.activation_based.auto_cuda import cfunction, base
from spikingjelly.activation_based.auto_cuda.neuron_kernel import *

class CuNeuronFPTTKernel(NeuronFPTTKernel):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.add_param(ctype=f'{dtype} *', cname='I_I_seq')

    def neuronal_charge(self) -> str:
        return '// neuronal_charge should be defined here!'


class CuNeuronBPTTKernel(NeuronBPTTKernel):
    def __init__(self, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        # self.add_param(ctype=f'const {dtype} &', cname='mem_decay')
        # self.add_param(ctype=f'const {dtype} &', cname='syn_decay')
        self.add_param(ctype=f'const {dtype} *', cname='grad_I_seq')
        self.add_param(ctype=f'const {dtype} *', cname='grad_h_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_I_init')

    def grad_h_next_to_v(self) -> str:
        return '// grad_h_next_to_v should be defined here!'

    
    def grad_I_next_to_I(self) -> str:
        return '// grad_I_next_to_I should be defined here!'
 
    def grad_h_next_to_I(self) -> str:
        return '// grad_h_next_to_I should be defined here!'
    
    def grad_I_to_x(self) -> str:
        return '// grad_I_next_to_x should be defined here!'


    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        if self.dtype == 'float':
            codes.append('float grad_h = 0.0f;')
            codes.append('float grad_I = 0.0f;')
            # codes.append('float grad_v = 0.0f;')
        elif self.dtype == 'half2':
            codes.append(cfunction.float2half2(y='half2 grad_h', x='0.0f'))
            codes.append(cfunction.float2half2(y='half2 grad_I', x='0.0f'))
            # codes.append(cfunction.float2half2(y='half2 grad_v', x='0.0f'))
        else:
            raise NotImplementedError(self.dtype)

        self._pre_core = codes.codes
        return self._pre_core

    @property
    def post_core(self):

        codes = base.CodeTyper(16)
        codes.append(self.grad_h_next_to_v())
        codes.append(cfunction.mul(z='grad_v_init[index]', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        codes.append(self.grad_I_next_to_I())
        codes.append(cfunction.mul(z='grad_I_init[index]', x='grad_I', y='grad_I_next_to_I', dtype=self.dtype))
        codes.append(self.grad_h_next_to_I())
        codes.append(cfunction.mul(z='grad_I', x='grad_h', y='grad_h_next_to_I', dtype=self.dtype))
        codes.append(cfunction.add(z='grad_I_init[index]', x='grad_I', y='grad_I_init[index]', dtype=self.dtype))
        self._post_core = codes.codes
        return self._post_core
    
    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h_seq[t]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} spike_seq_t', x='over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        # grad I with grad_I_seq
        with base.CodeBlock(core_codes):
            core_codes.append(self.grad_I_next_to_I()) #grad_I_seq[t] f'{self.dtype} grad_I_from_next'
            core_codes.append(cfunction.mul(z='grad_I', x='grad_I', y='grad_I_next_to_I', dtype=self.dtype))
            core_codes.append(self.grad_h_next_to_I())
            core_codes.append(cfunction.mul(z=f'{self.dtype} grad_I_from_h', x='grad_h', y='grad_h_next_to_I', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_I', x='grad_I', y='grad_I_from_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_I', x='grad_I', y='grad_I_seq[t]', dtype=self.dtype))
            # core_codes.append('grad_I_seq[t]=grad_I')
        
        # grad for x_seq
        core_codes.append(self.grad_I_to_x())
        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_I', y='grad_I_to_x', dtype=self.dtype))

        # grad_v2h
        if self.hard_reset:
            core_codes.append(
                cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
                              y='spike_seq_t', dtype=self.dtype))

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))
        else:
            core_codes.append(f'{self.dtype} grad_v_to_h = {cfunction.constant(None, 1., dtype=self.dtype)}')

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.sub(z=f'grad_v_to_h', x='grad_v_to_h', y='temp_var', dtype=self.dtype))

        # grad_h from (grad_v_seq + h_next_2v)
        core_codes.append(self.grad_h_next_to_v())
        core_codes.append(cfunction.mul(z=f'{self.dtype} grad_v', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_v', x='grad_v_seq[t]', y='grad_v', dtype=self.dtype))
        # core_codes.append('grad_v_seq[t]=grad_v')
        core_codes.append(cfunction.mul(z='grad_h', x='grad_v', y='grad_v_to_h', dtype=self.dtype))

        # grad_h from (grad_s_seq)
        with base.CodeBlock(core_codes):
            core_codes.append(
                cfunction.mul(z=f'{self.dtype} temp_var', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='temp_var', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='grad_h_seq[t]', dtype=self.dtype))
        # core_codes.append('grad_h_seq[t]=grad_h')

        self._core = core_codes.codes
        return self._core


class CuNeuronATGFBase:
    
    @staticmethod
    def pre_forward(py_dict: dict):
        device = py_dict['x_seq'].get_device()
        requires_grad = if_requires_grad(py_dict.values())
        scalar_to_cupy(py_dict)

        new_tensors(('h_seq', 'spike_seq', 'v_seq', 'I_seq'), py_dict)
        py_dict['v_v_seq'] = torch.cat((py_dict.pop('v_init').unsqueeze(0), py_dict.pop('v_seq')))
        py_dict['I_I_seq'] = torch.cat((py_dict.pop('I_init').unsqueeze(0), py_dict.pop('I_seq')))
        numel = py_dict['x_seq'].numel()
        N = py_dict['x_seq'].shape[1]
        threads = configure.cuda_threads
        if py_dict['x_seq'].dtype == torch.float16:
            # we will take two neurons to calculate as one neuron in cuda half2
            # pad will be implemented by the kernel.__call__
            N = math.ceil(N / 2)
            numel = N * py_dict['x_seq'].shape[0]

        blocks = cuda_utils.cal_blocks(N)

        with cuda_utils.DeviceEnvironment(device):
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)

        py_dict['numel'] = numel
        py_dict['N'] = N

        return requires_grad, blocks, threads, py_dict
    
    
    @staticmethod
    def pre_backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor,  grad_I_seq: torch.Tensor, grad_h_seq: torch.Tensor):
        backward_kernel = ctx.backward_kernel
        blocks = ctx.blocks
        threads = ctx.threads
        
        h_seq = ctx.saved_tensors[0]
        numel = ctx.numel
        N = ctx.N
        v_th = ctx.v_th
        v_reset = ctx.v_reset

        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 2
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -2]
        grad_v_init = zero_data[-2]
        grad_I_init = zero_data[-1] ##

        py_dict = {
            'numel': numel,
            'N': N,
            'grad_spike_seq': grad_spike_seq,
            'grad_v_seq': grad_v_seq,
            'grad_I_seq': grad_I_seq, ##
            'grad_h_seq': grad_h_seq, ##
            'h_seq': h_seq,
            'grad_x_seq': grad_x_seq,
            'grad_v_init': grad_v_init,
            'grad_I_init': grad_I_init, ##
            'v_th': v_th,
            'v_reset': v_reset
        }

        return backward_kernel, blocks, threads, py_dict


    @staticmethod
    def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
        """
        :param ctx: ``ctx`` in :class:`torch.autograd.Function`
        :param requires_grad: if any tensor in forward params requires grad
        :type requires_grad: bool
        :param args: tensors that need to be saved by ``ctx.save_for_backward``
        :param kwargs: items that need to be saved by ``ctx.xx = xx``

        Saves ``*args, **kwargs`` in ``ctx`` by ``ctx.save_for_backward(*args)`` and ``ctx.xx = xx`` for all ``xx`` in ``kwargs.items()``.
        """
        if requires_grad:
            ctx.save_for_backward(*args)
            for key, value in kwargs.items():
                ctx.__setattr__(key, value)

        
class CuLIFNodeFPTTKernel(CuNeuronFPTTKernel):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.add_param(ctype=f'const {dtype} &', cname='mem_decay')
        self.add_param(ctype=f'const {dtype} &', cname='syn_decay')
        # self.add_param(ctype=f'{dtype} *', cname='I_I_seq')


    def neuronal_charge(self) -> str:

        codes = cfunction.mul(z=f'{self.dtype} syn_temp_var', x='syn_decay', y='I_I_seq[t]',
                                   dtype=self.dtype)
        codes += cfunction.add(z='syn_temp_var', x='x_seq[t]', y='syn_temp_var', dtype=self.dtype)
        codes += f'I_I_seq[t+dt] = syn_temp_var;' #t+dt
        
        if self.hard_reset:
            codes += cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes += f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'

        
        codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='LIFNodeFPTTKernel_temp_var', y='I_I_seq[t]', dtype=self.dtype)
        codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='mem_decay', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='I_I_seq[t]', dtype=self.dtype)
        if self.hard_reset:
            codes += cfunction.add(z='h_seq[t]', x='h_seq[t]', y='v_reset', dtype=self.dtype)

        return codes


class CuLIFNodeBPTTKernel(CuNeuronBPTTKernel):
    def __init__(self, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.add_param(ctype=f'const {dtype} &', cname='mem_decay')
        self.add_param(ctype=f'const {dtype} &', cname='syn_decay')


    def grad_h_next_to_v(self) -> str:
        return f'const {self.dtype} grad_h_next_to_v = mem_decay;'
    
    def grad_I_next_to_I(self) -> str:
        return f'const {self.dtype} grad_I_next_to_I = syn_decay;'
 
    def grad_h_next_to_I(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_I', x=cfunction.constant(None, x=1., dtype=self.dtype), y='mem_decay', dtype=self.dtype)
    
    def grad_I_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_I_to_x', x=1., dtype=self.dtype)


class CuLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, I_init: torch.Tensor, v_th: float, v_reset: float or None,
                mem_decay: float, syn_decay: float, 
                forward_kernel: CuLIFNodeFPTTKernel, backward_kernel: CuLIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'I_init': I_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'mem_decay': mem_decay,
            'syn_decay': syn_decay,
        }
        
        requires_grad, blocks, threads, py_dict = CuNeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
            
        forward_kernel((blocks,), (threads,), py_dict)
        
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        CuNeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                        numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                        backward_kernel=backward_kernel, mem_decay=py_dict['mem_decay'], syn_decay=py_dict['syn_decay'])

        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ], py_dict['I_I_seq'][1:, ], py_dict['h_seq']

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_I_seq: torch.Tensor, grad_h_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = CuNeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq, grad_I_seq, grad_h_seq)
        py_dict['mem_decay'] = ctx.mem_decay
        py_dict['syn_decay'] = ctx.syn_decay

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None


        return py_dict['grad_x_seq'], py_dict['grad_v_init'], py_dict['grad_I_init'], None, None, None, None, None, None


class ParametricCuLIFNodeFPTTKernel(CuNeuronFPTTKernel):
    def __init__(self, decay_mode, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_mode = decay_mode
        self.add_param(ctype=f'const {dtype} *', cname='mem_decay')
        self.add_param(ctype=f'const {dtype} *', cname='syn_decay')
        
    def neuronal_charge(self) -> str:
        if self.decay_mode == 's':
            codes = cfunction.mul(z=f'{self.dtype} syn_temp_var', x='syn_decay[0]', y='I_I_seq[t]',
                                    dtype=self.dtype)
        else:
            codes = cfunction.mul(z=f'{self.dtype} syn_temp_var', x='syn_decay[index]', y='I_I_seq[t]',
                                    dtype=self.dtype)
        # codes += f'printf("%d, %d, %d \n", blockDim, blockIdx, threadIdx);'
        codes += cfunction.add(z='syn_temp_var', x='x_seq[t]', y='syn_temp_var', dtype=self.dtype)
        codes += f'I_I_seq[t+dt] = syn_temp_var;' #t+dt
        
        if self.hard_reset:
            codes += cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes += f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'

        
        codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='LIFNodeFPTTKernel_temp_var', y='I_I_seq[t]', dtype=self.dtype)
        if self.decay_mode == 's':
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='mem_decay[0]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='mem_decay[index]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            
        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='I_I_seq[t]', dtype=self.dtype)
        if self.hard_reset:
            codes += cfunction.add(z='h_seq[t]', x='h_seq[t]', y='v_reset', dtype=self.dtype)

        return codes


class ParametricCuLIFNodeBPTTKernel(CuNeuronBPTTKernel):
    def __init__(self, surrogate_function: Callable, decay_mode:str, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_mode = decay_mode
        self.add_param(ctype=f'const {dtype} *', cname='mem_decay')
        self.add_param(ctype=f'const {dtype} *', cname='syn_decay')
        
        # for param
        self.add_param(ctype=f'{dtype} *', cname='grad_mem_decay')
        self.add_param(ctype=f'{dtype} *', cname='grad_syn_decay')
        

        # self.add_param(ctype=f'const int &', cname='param_len')
        
        # float to avoid overflow
        self.add_param(ctype=f'const {dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'const {dtype} *', cname='I_I_seq')


    def grad_h_next_to_v(self) -> str:
        if self.decay_mode == 's':
            return f'const {self.dtype}  grad_h_next_to_v = mem_decay[0];'
        else: #self.decay_mode == 'm'
            return f'const {self.dtype}  grad_h_next_to_v = mem_decay[index];'
        
    def grad_I_next_to_I(self) -> str:
        if self.decay_mode == 's':
            return f'const {self.dtype}  grad_I_next_to_I = syn_decay[0];'
        else: #self.decay_mode == 'm'
            return f'const {self.dtype}  grad_I_next_to_I = syn_decay[index];'
        
 
    def grad_h_next_to_I(self) -> str:
        if self.decay_mode == 's':
            return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_I', x=cfunction.constant(None, x=1., dtype=self.dtype), y='mem_decay[0]', dtype=self.dtype)
        else:
            return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_I', x=cfunction.constant(None, x=1., dtype=self.dtype), y='mem_decay[index]', dtype=self.dtype)
    
    def grad_I_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_I_to_x', x=1., dtype=self.dtype)
            


    @property
    def head(self):
        # override
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        '''
        
        if self.decay_mode == 's':
            codes += fr'''
            __shared__ float sdata_mem[{configure.cuda_threads}];
            '''
            codes += fr'''
            __shared__ float sdata_syn[{configure.cuda_threads}];
            '''
        codes += '''
            if (index < N)
            {
                const int dt = N;
        '''

        codes += self.pre_core

        if self.reverse:
            codes += '''
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            '''
        else:
            codes += '''
                for(int t = index; t < numel; t += dt)
                {
            '''
        return codes


    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        # use float to avoid overflow
        if self.decay_mode == 's':
            codes.append('sdata_mem[threadIdx.x] = 0.0f;')
            codes.append('sdata_syn[threadIdx.x] = 0.0f;')
        return super().pre_core + '\n' + codes.codes

    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
            core_codes.append(cfunction.sub(z=f'{self.dtype} temp_var_mem', x='v_v_seq[t]', y='I_I_seq[t]', dtype=self.dtype))
            core_codes.append(cfunction.mul(z='temp_var_mem', x='temp_var_mem', y='grad_h', dtype=self.dtype))

            core_codes.append(cfunction.mul(z=f'{self.dtype} temp_var_syn', x='I_I_seq[t]', y='grad_I', dtype=self.dtype))
            
            if self.decay_mode == 'm':
                core_codes.append('grad_mem_decay[index] += temp_var_mem;')
                core_codes.append('grad_syn_decay[index] += temp_var_syn;')
            else:  
                if self.dtype == 'float':
                    core_codes.append('sdata_mem[threadIdx.x] += temp_var_mem;')
                    core_codes.append('sdata_syn[threadIdx.x] += temp_var_syn;')
                elif self.dtype == 'half2':
                    core_codes.append('sdata_mem[threadIdx.x] += __half2float(__hadd(__low2half(temp_var_mem), __high2half(temp_var_mem)));')
                    core_codes.append('sdata_syn[threadIdx.x] += __half2float(__hadd(__low2half(temp_var_syn), __high2half(temp_var_syn)));')
                else:
                    raise NotImplementedError(self.dtype)

        return super().core + '\n' + core_codes.codes

    @property
    def tail(self):
        codes = '''
                }
        '''

        codes += self.post_core

        if self.decay_mode == 's':
            codes += '''
            }
            else
            {
                sdata_mem[threadIdx.x] = 0.0f;
                sdata_syn[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < stride)
            {
                sdata_mem[threadIdx.x] += sdata_mem[threadIdx.x + stride];
                sdata_syn[threadIdx.x] += sdata_syn[threadIdx.x + stride];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                atomicAdd(grad_mem_decay, sdata_mem[0]);
                atomicAdd(grad_syn_decay, sdata_syn[0]);
            }    
        }
        '''
        else:
            codes += '''
                }
            }
            '''
        return codes


class ParametricCuLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, I_init: torch.Tensor, v_th: float, v_reset: float or None,
                mem_decay: torch.Tensor, syn_decay: torch.Tensor, output_size: int, decay_mode: str,
                forward_kernel: ParametricCuLIFNodeFPTTKernel, backward_kernel: ParametricCuLIFNodeBPTTKernel):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError('When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!')
        
        
        if decay_mode == 'm':
            # batch_size = x_seq.shape[1]//mem_decay.shape[-1]
            batch_size = x_seq.shape[1] // output_size
            mem_decay = mem_decay.repeat(batch_size) # BxC
            syn_decay = syn_decay.repeat(batch_size) # BxC
        
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'I_init': I_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'mem_decay': mem_decay,
            'syn_decay': syn_decay,
            # 'decay_mode': decay_mode,
        }

        requires_grad, blocks, threads, py_dict = CuNeuronATGFBase.pre_forward(py_dict)
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
            
        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        CuNeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], py_dict['v_v_seq'], py_dict['I_I_seq'], blocks=blocks, threads=threads,
                        numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                        backward_kernel=backward_kernel, mem_decay=py_dict['mem_decay'], syn_decay=py_dict['syn_decay'], \
                            output_size=output_size, decay_mode=decay_mode)

        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ], py_dict['I_I_seq'][1:, ], py_dict['h_seq']


    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_I_seq: torch.Tensor, grad_h_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = CuNeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq, grad_I_seq, grad_h_seq)
        py_dict['mem_decay'] = ctx.mem_decay
        py_dict['syn_decay'] = ctx.syn_decay
        py_dict['grad_mem_decay'] = torch.zeros_like(ctx.mem_decay, dtype=torch.float)
        py_dict['grad_syn_decay'] = torch.zeros_like(ctx.syn_decay, dtype=torch.float)
        # py_dict['param_len'] = int(ctx.syn_decay.shape[-1])
        # print(py_dict['param_len'])
        py_dict['v_v_seq'] = ctx.saved_tensors[1]
        py_dict['I_I_seq'] = ctx.saved_tensors[2]

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)
        if ctx.decay_mode == 'm':
            py_dict['grad_mem_decay'] = py_dict['grad_mem_decay'].reshape(-1, ctx.output_size).sum(0)
            py_dict['grad_syn_decay'] = py_dict['grad_syn_decay'].reshape(-1, ctx.output_size).sum(0)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None


        return py_dict['grad_x_seq'], py_dict['grad_v_init'], py_dict['grad_I_init'], None, None, py_dict['grad_mem_decay'], py_dict['grad_syn_decay'], None, None, None, None



class AdLIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_mode, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_mode = decay_mode
        self.add_param(ctype=f'const {dtype} *', cname='mem_decay')
        self.add_param(ctype=f'const {dtype} *', cname='osc_decay')
        self.add_param(ctype=f'const {dtype} *', cname='a')
        self.add_param(ctype=f'const {dtype} *', cname='b')
        self.add_param(ctype=f'{dtype} *', cname='w_w_seq')
        self.add_param(ctype=f'{dtype} *', cname='h_h_seq')
        self.add_param(ctype=f'{dtype} *', cname='s_s_seq')

        self.cparams.pop('h_seq')
        self.cparams.pop('spike_seq')


    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes += f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'

        
        codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='LIFNodeFPTTKernel_temp_var', y='x_seq[t]', dtype=self.dtype)
        codes += cfunction.add(z='LIFNodeFPTTKernel_temp_var', x='LIFNodeFPTTKernel_temp_var', y='w_w_seq[t]', dtype=self.dtype)
        if self.decay_mode == 's':
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='mem_decay[0]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='mem_decay[index]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            
        codes += cfunction.add(z='LIFNodeFPTTKernel_temp_var', x='LIFNodeFPTTKernel_temp_var', y='x_seq[t]', dtype=self.dtype)
        codes += cfunction.sub(z='h_h_seq[t+dt]', x='LIFNodeFPTTKernel_temp_var', y='w_w_seq[t]', dtype=self.dtype)
        
        if self.hard_reset:
            codes += cfunction.add(z='h_h_seq[t+dt]', x='h_h_seq[t+dt]', y='v_reset', dtype=self.dtype)
        
        if self.decay_mode == 's':
            codes += cfunction.mul(z=f'{self.dtype} osc_temp_var', x='osc_decay[0]', y='w_w_seq[t]',
                                    dtype=self.dtype)
            codes += cfunction.mul(z=f'{self.dtype} osc_temp_var1', x='a[0]', y='h_h_seq[t]',
                                    dtype=self.dtype)
            codes += cfunction.mul(z=f'{self.dtype} osc_temp_var2', x='b[0]', y='s_s_seq[t]',
                                    dtype=self.dtype)
        else:
            codes += cfunction.mul(z=f'{self.dtype} osc_temp_var', x='osc_decay[index]', y='w_w_seq[t]',
                                    dtype=self.dtype)
            codes += cfunction.mul(z=f'{self.dtype} osc_temp_var1', x='a[index]', y='h_h_seq[t]',
                                    dtype=self.dtype)
            codes += cfunction.mul(z=f'{self.dtype} osc_temp_var2', x='b[index]', y='s_s_seq[t]',
                                    dtype=self.dtype)
        codes += cfunction.add(z='w_w_seq[t+dt]', x='osc_temp_var', y='osc_temp_var1', dtype=self.dtype)
        codes += cfunction.add(z='w_w_seq[t+dt]', x='w_w_seq[t+dt]', y='osc_temp_var2', dtype=self.dtype)
        
        # codes += 'printf("%d, %f \n",t, h_h_seq[t+dt])'
        # codes += 'printf("%d, %f \n",t, h_h_seq[t+dt])'
        return codes

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())
        
        core_codes.append(neuronal_fire(spike='s_s_seq[t+dt]', v='h_h_seq[t+dt]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(v_next='v_v_seq[t + dt]', h='h_h_seq[t+dt]', spike='s_s_seq[t+dt]', v_reset='v_reset',
                                    dtype=self.dtype))
        else:
            core_codes.append(
                neuronal_soft_reset(v_next='v_v_seq[t + dt]', h='h_h_seq[t+dt]', spike='s_s_seq[t+dt]', v_th='v_th',
                                    dtype=self.dtype))

        self._core = core_codes.codes
        # return super()._core + '\n' + codes.codes
        return self._core


class AdLIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self, surrogate_function: Callable, decay_mode:str, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_mode = decay_mode
        
        self.add_param(ctype=f'{dtype} *', cname='grad_w_init')
        
        self.add_param(ctype=f'const {dtype} *', cname='mem_decay')
        self.add_param(ctype=f'const {dtype} *', cname='osc_decay')
        self.add_param(ctype=f'const {dtype} *', cname='a')
        self.add_param(ctype=f'const {dtype} *', cname='b')
        
        # for param
        self.add_param(ctype=f'const {dtype} *', cname='grad_h_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_mem_decay')
        self.add_param(ctype=f'{dtype} *', cname='grad_osc_decay')
        self.add_param(ctype=f'{dtype} *', cname='grad_a')
        self.add_param(ctype=f'{dtype} *', cname='grad_b')
        

        # float to avoid overflow
        self.add_param(ctype=f'const {dtype} *', cname='x_seq')
        self.add_param(ctype=f'const {dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'const {dtype} *', cname='w_w_seq')
        self.add_param(ctype=f'const {dtype} *', cname='s_s_seq')
        self.add_param(ctype=f'const {dtype} *', cname='h_h_seq')
  

    def grad_h_next_to_v(self) -> str:
        if self.decay_mode == 's':
            return f'const {self.dtype}  grad_h_next_to_v = mem_decay[0];'
        else: #self.decay_mode == 'm'
            return f'const {self.dtype}  grad_h_next_to_v = mem_decay[index];'
 
    def grad_h_to_x(self) -> str:
        if self.decay_mode == 's':
            return cfunction.sub(z=f'const {self.dtype} grad_h_to_x', x=cfunction.constant(None, x=1., dtype=self.dtype), y='mem_decay[0]', dtype=self.dtype)
        else:
            return cfunction.sub(z=f'const {self.dtype} grad_h_to_x', x=cfunction.constant(None, x=1., dtype=self.dtype), y='mem_decay[index]', dtype=self.dtype)
    
    def grad_h_next_to_w(self) -> str:
        if self.decay_mode == 's':
            return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_w', x='mem_decay[0]', y=cfunction.constant(None, x=1., dtype=self.dtype), dtype=self.dtype)
        else:
            return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_w', x='mem_decay[index]', y=cfunction.constant(None, x=1., dtype=self.dtype), dtype=self.dtype)
    
    def grad_w_next_to_w(self) -> str:
        if self.decay_mode == 's':
            return f'const {self.dtype}  grad_w_next_to_w = osc_decay[0];'
        else: #self.decay_mode == 'm'
            return f'const {self.dtype}  grad_w_next_to_w = osc_decay[index];'
        
    def grad_w_next_to_h(self) -> str:
        if self.decay_mode == 's':
            return f'const {self.dtype}  grad_w_next_to_h = a[0];'
        else: #self.decay_mode == 'm'
            return f'const {self.dtype}  grad_w_next_to_h = a[index];'
    
    def grad_w_next_to_s(self) -> str:
        if self.decay_mode == 's':
            return f'const {self.dtype}  grad_w_next_to_s = b[0];'
        else: #self.decay_mode == 'm'
            return f'const {self.dtype}  grad_w_next_to_s = b[index];'
        

    @property
    def head(self):
        # override
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        '''
        
        if self.decay_mode == 's':
            codes += fr'''
            __shared__ float sdata_mem[{configure.cuda_threads}];
            '''
            codes += fr'''
            __shared__ float sdata_osc[{configure.cuda_threads}];
            '''
            codes += fr'''
            __shared__ float sdata_a[{configure.cuda_threads}];
            '''
            codes += fr'''
            __shared__ float sdata_b[{configure.cuda_threads}];
            '''
        codes += '''
            if (index < N)
            {
                const int dt = N;
        '''

        codes += self.pre_core

        if self.reverse:
            codes += '''
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            '''
        else:
            codes += '''
                for(int t = index; t < numel; t += dt)
                {
            '''
        return codes
    
    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        # use float to avoid overflow
        if self.dtype == 'float':
            codes.append('float grad_h = 0.0f;')
            codes.append('float grad_w = 0.0f;')
        elif self.dtype == 'half2':
            codes.append(cfunction.float2half2(y='half2 grad_h', x='0.0f'))
            codes.append(cfunction.float2half2(y='half2 grad_w', x='0.0f'))
        else:
            raise NotImplementedError(self.dtype)
        
        if self.decay_mode == 's':
            codes.append('sdata_mem[threadIdx.x] = 0.0f;')
            codes.append('sdata_osc[threadIdx.x] = 0.0f;')
            codes.append('sdata_a[threadIdx.x] = 0.0f;')
            codes.append('sdata_b[threadIdx.x] = 0.0f;')
            
        return codes.codes #super().pre_core + '\n' + 

    @property
    def post_core(self):
        codes = base.CodeTyper(16)
        codes.append(self.grad_w_next_to_w())
        codes.append(cfunction.mul(z='grad_w_init[index]', x='grad_w', y='grad_w_next_to_w', dtype=self.dtype))
        codes.append(self.grad_h_next_to_w())
        codes.append(cfunction.mul(z='grad_w', x='grad_h', y='grad_h_next_to_w', dtype=self.dtype))
        codes.append(cfunction.add(z='grad_w_init[index]', x='grad_w', y='grad_w_init[index]', dtype=self.dtype))
        
        return super().post_core  + '\n' + codes.codes
    
    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        # compute grad_w
        core_codes.append(f'{self.dtype} grad_w_next = grad_w')
        # core_codes.append('grad_w_seq[t] = grad_w')
        core_codes.append(self.grad_w_next_to_w())
        core_codes.append(cfunction.mul(z='grad_w', x='grad_w_next', y='grad_w_next_to_w', dtype=self.dtype))
        core_codes.append(self.grad_h_next_to_w())
        core_codes.append(cfunction.mul(z=f'const {self.dtype} grad_w_tmp', x='grad_h', y='grad_h_next_to_w', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_w', x='grad_w', y='grad_w_tmp', dtype=self.dtype))
        
        

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h_h_seq[t+dt]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} spike_seq_t', x='over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        # grad h from v
        if self.hard_reset:
            core_codes.append(
                cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
                              y='spike_seq_t', dtype=self.dtype))

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h_h_seq[t+dt]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))
        else:
            core_codes.append(f'{self.dtype} grad_v_to_h = {cfunction.constant(None, 1., dtype=self.dtype)}')

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.sub(z=f'grad_v_to_h', x='grad_v_to_h', y='temp_var', dtype=self.dtype))

        # grad_h
        core_codes.append(self.grad_h_next_to_v())
        core_codes.append(cfunction.mul(z=f'{self.dtype} grad_v', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_v', x='grad_v_seq[t]', y='grad_v', dtype=self.dtype))
        # core_codes.append('grad_v_seq[t]=grad_v')
        core_codes.append(cfunction.mul(z='grad_h', x='grad_v', y='grad_v_to_h', dtype=self.dtype))

        # grad h from s
        with base.CodeBlock(core_codes):
            core_codes.append(self.grad_w_next_to_s())
            core_codes.append(
                cfunction.mul(z=f'{self.dtype} grad_s', x='grad_w_next', y='grad_w_next_to_s', dtype=self.dtype)) #grad_w_next
            core_codes.append(
                cfunction.add(z='grad_s', x='grad_s', y='grad_spike_seq[t]', dtype=self.dtype))
            core_codes.append(
                cfunction.mul(z='grad_s', x='grad_s', y='grad_s_to_h', dtype=self.dtype))
            # core_codes.append(
            #     cfunction.mul(z=f'{self.dtype} grad_s', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='grad_s', dtype=self.dtype))
        
        # grad h from w
        ###########
        core_codes.append(self.grad_w_next_to_h())
        core_codes.append(cfunction.mul(z=f'{self.dtype} grad_h_from_w', x='grad_w_next', y='grad_w_next_to_h', dtype=self.dtype)) #_next
        core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='grad_h_from_w', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='grad_h_seq[t]', dtype=self.dtype))
        
        # grad x
        core_codes.append(self.grad_h_to_x())
        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))

        with base.CodeBlock(core_codes):
            core_codes.append(cfunction.sub(z=f'{self.dtype} temp_var_mem', x='v_v_seq[t]', y='x_seq[t]', dtype=self.dtype))
            core_codes.append(cfunction.add(z='temp_var_mem', x='temp_var_mem', y='w_w_seq[t]', dtype=self.dtype))
            core_codes.append(cfunction.mul(z='temp_var_mem', x='temp_var_mem', y='grad_h', dtype=self.dtype))

            core_codes.append(cfunction.mul(z=f'{self.dtype} temp_var_osc', x='w_w_seq[t]', y='grad_w', dtype=self.dtype)) #grad_w_next
            core_codes.append(cfunction.mul(z=f'{self.dtype} temp_var_a', x='h_h_seq[t]', y='grad_w', dtype=self.dtype)) #grad_w_next
            core_codes.append(cfunction.mul(z=f'{self.dtype} temp_var_b', x='s_s_seq[t]', y='grad_w', dtype=self.dtype))

            if self.decay_mode == 'm':
                core_codes.append('grad_mem_decay[index] += temp_var_mem;')
                core_codes.append('grad_osc_decay[index] += temp_var_osc;')
                core_codes.append('grad_a[index] += temp_var_a;')
                core_codes.append('grad_b[index] += temp_var_b;')
            else:  
                if self.dtype == 'float':
                    core_codes.append('sdata_mem[threadIdx.x] += temp_var_mem;')
                    core_codes.append('sdata_osc[threadIdx.x] += temp_var_osc;')
                    core_codes.append('sdata_a[threadIdx.x] += temp_var_a;')
                    core_codes.append('sdata_b[threadIdx.x] += temp_var_b;')
                elif self.dtype == 'half2':
                    core_codes.append('sdata_mem[threadIdx.x] += __half2float(__hadd(__low2half(temp_var_mem), __high2half(temp_var_mem)));')
                    core_codes.append('sdata_osc[threadIdx.x] += __half2float(__hadd(__low2half(temp_var_osc), __high2half(temp_var_osc)));')
                    core_codes.append('sdata_a[threadIdx.x] += __half2float(__hadd(__low2half(temp_var_a), __high2half(temp_var_a)));')
                    core_codes.append('sdata_b[threadIdx.x] += __half2float(__hadd(__low2half(temp_var_b), __high2half(temp_var_b)));')
                else:
                    raise NotImplementedError(self.dtype)

        self._core = core_codes.codes
        return self._core


    @property
    def tail(self):
        codes = '''
                }
        '''

        codes += self.post_core

        if self.decay_mode == 's':
            codes += '''
            }
            else
            {
                sdata_mem[threadIdx.x] = 0.0f;
                sdata_osc[threadIdx.x] = 0.0f;
                sdata_a[threadIdx.x] = 0.0f;
                sdata_b[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < stride)
            {
                sdata_mem[threadIdx.x] += sdata_mem[threadIdx.x + stride];
                sdata_osc[threadIdx.x] += sdata_osc[threadIdx.x + stride];
                sdata_a[threadIdx.x] += sdata_a[threadIdx.x + stride];
                sdata_b[threadIdx.x] += sdata_b[threadIdx.x + stride];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                atomicAdd(grad_mem_decay, sdata_mem[0]);
                atomicAdd(grad_osc_decay, sdata_osc[0]);
                atomicAdd(grad_a, sdata_a[0]);
                atomicAdd(grad_b, sdata_b[0]);
            }    
        }
        '''
        else:
            codes += '''
                }
            }
            '''
        return codes


class AdLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, w_init: torch.Tensor, v_th: float, v_reset: float or None,
                mem_decay: torch.Tensor, osc_decay: torch.Tensor, a: torch.Tensor, b: torch.Tensor, 
                output_size: int, decay_mode: str,
                forward_kernel: AdLIFNodeFPTTKernel, backward_kernel: AdLIFNodeBPTTKernel):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError('When using the the neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!')
        
        if decay_mode == 'm':
            # batch_size = x_seq.shape[1]//mem_decay.shape[-1]
            batch_size = x_seq.shape[1] // output_size
            mem_decay = mem_decay.repeat(batch_size) # BxC
            osc_decay = osc_decay.repeat(batch_size) # BxC
            a = a.repeat(batch_size) # BxC
            b = b.repeat(batch_size) # BxC
            
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'w_init': w_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'mem_decay': mem_decay,
            'osc_decay': osc_decay,
            'a': a,
            'b': b,
            # 'decay_mode': decay_mode,
        }

        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
        
        zero_init = torch.zeros_like(py_dict['w_init'])
        py_dict['w_seq'] = torch.zeros_like(py_dict['h_seq'])
        py_dict['w_w_seq'] = torch.cat((py_dict.pop('w_init').unsqueeze(0), py_dict.pop('w_seq')))
        py_dict['h_h_seq'] = torch.cat((zero_init.unsqueeze(0), py_dict.pop('h_seq')))
        py_dict['s_s_seq'] = torch.cat((zero_init.unsqueeze(0), py_dict.pop('spike_seq')))
        # print(py_dict['mem_decay'],py_dict['osc_decay'],  py_dict['a'],  py_dict['b'] )
        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        # print(py_dict['s_s_seq'].shape,py_dict['v_v_seq'],  py_dict['h_h_seq'] )
        NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['x_seq'], py_dict['s_s_seq'], py_dict['h_h_seq'], py_dict['v_v_seq'], py_dict['w_w_seq'], blocks=blocks, threads=threads,
                        numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                        backward_kernel=backward_kernel, mem_decay=py_dict['mem_decay'], 
                        osc_decay=py_dict['osc_decay'], a=py_dict['a'], b=py_dict['b'],
                        output_size=output_size, decay_mode=decay_mode)
        
        
        return py_dict['s_s_seq'][1:,], py_dict['v_v_seq'][1:, ],  py_dict['h_h_seq'][1:,]


    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_h_seq: torch.Tensor):
        
        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['x_seq'] = ctx.saved_tensors[0]
        py_dict['s_s_seq'] = ctx.saved_tensors[1]
        py_dict['h_h_seq'] = ctx.saved_tensors[2]
        py_dict['v_v_seq'] = ctx.saved_tensors[3]
        py_dict['w_w_seq'] = ctx.saved_tensors[4]
        py_dict['mem_decay'] = ctx.mem_decay
        py_dict['osc_decay'] = ctx.osc_decay
        py_dict['a'] = ctx.a
        py_dict['b'] = ctx.b
        py_dict['grad_h_seq'] = grad_h_seq ####
        py_dict['grad_mem_decay'] = torch.zeros_like(ctx.mem_decay, dtype=torch.float)
        py_dict['grad_osc_decay'] = torch.zeros_like(ctx.osc_decay, dtype=torch.float)
        py_dict['grad_a'] = torch.zeros_like(ctx.a, dtype=torch.float)
        py_dict['grad_b'] = torch.zeros_like(ctx.b, dtype=torch.float)
        
        py_dict['grad_w_init'] = torch.zeros_like(py_dict['grad_v_init'])
 
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if ctx.decay_mode == 'm':
            py_dict['grad_mem_decay'] = py_dict['grad_mem_decay'].reshape(-1, ctx.output_size).sum(0)
            py_dict['grad_osc_decay'] = py_dict['grad_osc_decay'].reshape(-1, ctx.output_size).sum(0)
            py_dict['grad_a'] = py_dict['grad_a'].reshape(-1, ctx.output_size).sum(0)
            py_dict['grad_b'] = py_dict['grad_b'].reshape(-1, ctx.output_size).sum(0)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None
        
        return py_dict['grad_x_seq'], py_dict['grad_v_init'], py_dict['grad_w_init'], None, None, py_dict['grad_mem_decay'], py_dict['grad_osc_decay'], py_dict['grad_a'], py_dict['grad_b'], None, None, None, None


