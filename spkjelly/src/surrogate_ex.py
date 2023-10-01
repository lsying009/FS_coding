import torch
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase, heaviside
from spikingjelly.activation_based.auto_cuda import cfunction


@torch.jit.script
def mysoftsign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = grad_output / (alpha * x.abs() + 1.0) ** 2
    return sgax, None


class mysoftsign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return mysoftsign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class MySoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=5.0, spiking=True):
        super().__init__(alpha, spiking)
        # spiking == True, useless
    
    @staticmethod
    def spiking_function(x, alpha):
        return mysoftsign.apply(x, alpha)
    
    @staticmethod
    def backward(grad_output, x, alpha):
        return mysoftsign_backward(grad_output, x, alpha)[0]
    
    def forward(self, x: torch.Tensor):
        return self.spiking_function(x, self.alpha)

    def backward_cuda_codes(y: str, x: str, alpha: float, dtype: str):
        alpha = cfunction.constant(None, alpha, dtype)
        one = cfunction.constant(None, 1.0, dtype)
        two = cfunction.constant(None, 2.0, dtype)
        codes = cfunction.abs(y=f'{dtype} sugd_backward_temp', x=x, dtype=dtype)
        codes += cfunction.mul(z='sugd_backward_temp', x='sugd_backward_temp', y=alpha, dtype=dtype)
        codes += cfunction.add(z='sugd_backward_temp', x='sugd_backward_temp', y=one, dtype=dtype)
        if dtype == 'float':
            codes += f'sugd_backward_temp = __powf(sugd_backward_temp, {two});'
        elif dtype == 'half2':
            # CUDA FP16 does not provide powf function. We use z = 2 ** (log2(x) * y)
            codes += f'sugd_backward_temp = h2exp(__hmul2(h2log2(sugd_backward_temp), {two}));'
        # codes += cfunction.power(z='sugd_backward_temp', x='sugd_backward_temp', y='2.0f', dtype=dtype)
        codes += cfunction.div(z=y, x=one, y='sugd_backward_temp', dtype=dtype)
        return codes
    
    def cuda_codes(self, y: str, x: str, dtype: str):
        return MySoftSign.backward_cuda_codes(y=y, x=x, alpha=self.alpha, dtype=dtype)
  
  
