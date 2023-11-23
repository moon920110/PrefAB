import math
import torch
import torch.nn.init as init
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoModel

from network.autoencoder import AutoEncoder


class RankNet(nn.Module):
    def __init__(self, config):
        super(RankNet, self).__init__()
        self.config = config

        f_dim = config['train']['f_dim']
        if config['train']['base_transformer_model'] == 'bert':
            self.transformer_encoder = AutoModel.from_pretrained('bert-base-uncased')
            d_model = self.transformer_encoder.config.hidden_size
        else:
            d_model = config['train']['d_model']
            encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=config['train']['dropout'], batch_first=True)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)

        self.autoencoder = AutoEncoder()

        self.extractor = nn.Sequential(
            nn.Linear(f_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(),
            nn.Dropout1d(config['train']['dropout']),
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Dropout1d(config['train']['dropout']),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout1d(config['train']['dropout']),
            nn.Linear(2048, d_model),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.Dropout1d(config['train']['dropout']),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=config['train']['dropout'])

        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        if config['train']['base_transformer_model'] == 'Built-in':
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

    def forward(self, img1, input1, img2, input2, mask):
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

        if self.config['train']['base_transformer_model'] == 'Bert':
            x1 = self.transformer_encoder(inputs_embeds=input1, attention_mask=mask).pooler_output
            x2 = self.transformer_encoder(inputs_embeds=input2, attention_mask=mask).pooler_output
            x1 = self.fc(x1)
            x2 = self.fc(x2)
        else:
            x1 = self.transformer_encoder(input1)
            x2 = self.transformer_encoder(input2)
            avg_pooled1 = torch.mean(x1, dim=1)
            avg_pooled2 = torch.mean(x2, dim=1)
            x1 = self.fc(avg_pooled1)
            x2 = self.fc(avg_pooled2)
        return torch.log_softmax(x1 - x2, dim=-1), d1, d2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2 means 0, 2, 4, 6, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2 means 1, 3, 5, 7, ...

        if self.batch_first:
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        else:
            pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # register_buffer is not a parameter, but it is part of state_dict

    def forward(self, x):
        if self.batch_first:
            # x is (batch, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]
        else:
            # x is (seq_len, batch, d_model)
            x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)
