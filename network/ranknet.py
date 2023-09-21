import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import timm


class MLPBase(nn.Module):
    """
        net_structures: RankNet base network Layers (e.g., [input_dim, 64, 16, 1])
        double_precision: whether the tensor type is double or not
    """
    def __init__(self, net_structures, double_precision=False):
        super(MLPBase, self).__init__()

        self.fc_layers = len(net_structures)
        for i in range(self.fc_layers - 1):
            layer = nn.Linear(net_structures[i], net_structures[i+1])
            if double_precision:
                layer = layer.double()
            setattr(self, f'fc{i+1}', layer)  # flexibly generate and init fc layers

        self.activation = nn.ReLU()

    def forward(self, x):
        for i in range(1, self.fc_layers):
            fc = getattr(self, f'fc{i}')
            x = self.activation(fc(x))

        return x

    def dump_param(self):
        for i in range(1, self.fc_layers):
            print(f'fc{i} layers')
            fc = getattr(self, f'fc{i}')

            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(fc.bias).item(), torch.norm(fc.bias.grad).item()

            weight_ratio = weight_grad_norm / weight_norm if weight_norm else float('inf') if weight_grad_norm else 0.0
            bias_ratio = bias_grad_norm / bias_norm if bias_norm else float('inf') if bias_grad_norm else 0.0

            print(f'\tweight norm {weight_norm:.4e}, grad norm {weight_grad_norm:.4e}, ratio {weight_ratio:.4e}')
            print(f'\tbias norm {bias_norm:.4e}, grad norm {bias_grad_norm:.4e}, ratio {bias_ratio:.4e}')


class RankNet(nn.Module):
    def __init__(self, net_structures, double_precision=False):
        super(RankNet, self).__init__(net_structures, double_precision)

        self.base_network = MLPBase(net_structures, double_precision)

    def forward(self, input1, input2):
        x1 = self.base_network(input1)
        x2 = self.base_network(input2)
        return torch.sigmoid(x1 - x2)

    def dump_param(self):
        self.base_network.dump_param()


