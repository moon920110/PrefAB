import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


from network.autoencoder import AutoEncoder


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
    def __init__(self, config):
        super(RankNet, self).__init__()

        f_dim = config['train']['f_dim']
        d_model = config['train']['d_model']
        self.autoencoder = AutoEncoder()
        self.extractor = nn.Sequential(
            nn.Linear(f_dim, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, d_model),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=config['train']['dropout'])
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=config['train']['dropout'], batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        self.fc = nn.Linear(d_model, 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc:
                    init.xavier_normal_(m.weight)
                else:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)

    def forward(self, img1, input1, img2, input2):
        reshaped_img1 = img1.view(-1, *img1.shape[2:])  # batch, 4, c, h, w => batch * 4, c, h, w
        reshaped_img2 = img2.view(-1, *img2.shape[2:])
        e1, d1 = self.autoencoder(reshaped_img1)
        e2, d2 = self.autoencoder(reshaped_img2)
        e1 = e1.view(int(e1.shape[0]/4), 4, -1)
        e2 = e2.view(int(e2.shape[0]/4), 4, -1)

        input1 = torch.cat((input1, e1), dim=2)
        input1 = self.extractor(input1.view(-1, input1.shape[-1]))
        input1 = input1.view(int(input1.shape[0]/4), 4, -1)
        input2 = torch.cat((input2, e2), dim=2)
        input2 = self.extractor(input2.view(-1, input2.shape[-1]))
        input2 = input2.view(int(input2.shape[0]/4), 4, -1)

        input1 = self.pos_encoder(input1)
        input2 = self.pos_encoder(input2)

        x1 = self.transformer_encoder(input1)
        x2 = self.transformer_encoder(input2)
        avg_pooled1 = torch.mean(x1, dim=1)
        avg_pooled2 = torch.mean(x2, dim=1)
        x1 = self.fc(avg_pooled1)
        x2 = self.fc(avg_pooled2)
        return F.log_softmax(x1 - x2, dim=-1), d1, d2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2 means 0, 2, 4, 6, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2 means 1, 3, 5, 7, ...

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # register_buffer is not a parameter, but it is part of state_dict

    def forward(self, x):
        # x is (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
