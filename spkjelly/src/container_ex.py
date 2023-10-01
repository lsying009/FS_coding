
import torch
import torch.nn as nn

from spikingjelly.activation_based import base


class AddLinearRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, out_features: int, bias: bool = True,
                 step_mode='m') -> None:
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module_out_features = out_features
        # self.rc = layer.Linear(self.sub_module_out_features + in_features, self.sub_module_out_features, bias)
        self.rc = nn.Linear(self.sub_module_out_features, self.sub_module_out_features, bias)
        self.sub_module = sub_module
        self.register_memory('y', None)

    def single_step_forward(self, x: torch.Tensor):
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.zeros([x.shape[0], self.sub_module_out_features]).to(x)
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.sub_module_out_features)
                self.y = torch.zeros(out_shape).to(x)
        
        x = self.sub_module[:-1](x) + self.rc(self.y)
        self.y = self.sub_module[-1](x)
        return self.y

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        T = x_seq.shape[0]
        y_seq = []

        input_x_seq = self.sub_module[:-1](x_seq)
        if x_seq[0].ndim == 2:
            self.y = torch.zeros([x_seq[0].shape[0], self.sub_module_out_features]).to(x_seq)
        else:
            out_shape = [x_seq[0].shape[0]]
            out_shape.extend(x_seq[0].shape[1:-1])
            out_shape.append(self.sub_module_out_features)
            self.y = torch.zeros(out_shape).to(x_seq)

        for t in range(T):
            self.y = self.sub_module[-1](input_x_seq[t] + self.rc(self.y))
            y_seq.append(self.y.unsqueeze(0))

        return torch.cat(y_seq, 0)


    def extra_repr(self) -> str:
        return f', step_mode={self.step_mode}'

