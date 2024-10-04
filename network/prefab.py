import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from network.autoencoder import AutoEncoder


class Prefab(nn.Module):
    def __init__(self, config, meta_feature_size, bio_feature_size):
        super(Prefab, self).__init__()
        self.config = config
        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        ext_layers = []
        fc_layers = []
        bio_layers = []
        self.genre_size = bio_feature_size['genre']
        self.gender_size = bio_feature_size['gender']
        self.play_freq_size = bio_feature_size['play_freq']
        self.gamer_size = bio_feature_size['gamer']

        bio_embedding_dim = config['train']['bio_embedding_dim']
        f_dim = config['train']['f_dim']

        # bio extractor setup
        self.age_embedding = nn.Linear(1, bio_embedding_dim)
        self.genre_embedding = nn.Embedding(self.genre_size, bio_embedding_dim)
        bio_total_size = (
                self.gender_size +
                self.play_freq_size +
                self.gamer_size +
                3 +   # platforms (pc, mobile, console)
                2 * bio_embedding_dim
        )
        b_dim = bio_total_size
        while b_dim // 2 > bio_embedding_dim:
            bio_layers.append(nn.Linear(b_dim, b_dim//2))
            bio_layers.append(nn.LayerNorm(b_dim//2))
            bio_layers.append(nn.LeakyReLU())
            b_dim = b_dim // 2
        bio_layers.append(nn.Linear(b_dim, bio_embedding_dim))
        bio_layers.append(nn.LayerNorm(bio_embedding_dim))
        bio_layers.append(nn.LeakyReLU())

        self.bio_extractor = nn.Sequential(*bio_layers)

        # transformer setup
        d_model = config['train']['d_model']
        self.pos_encoder = PositionalEncoding(d_model, dropout=config['train']['dropout'])
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=config['train']['dropout'], batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.config['train']['num_transform_layers'])

        # autoencoder and feature extractor setup
        if self.mode == 'prefab' or self.mode == 'non_ordinal':
            self.autoencoder = AutoEncoder()

            while f_dim // 2 > d_model:
                ext_layers.append(nn.Linear(f_dim, f_dim//2))
                ext_layers.append(nn.LayerNorm(f_dim//2))
                ext_layers.append(nn.LeakyReLU())
                f_dim = f_dim // 2
            ext_layers.append(nn.Linear(f_dim, d_model - meta_feature_size - bio_embedding_dim))
            ext_layers.append(nn.LayerNorm(d_model - meta_feature_size - bio_embedding_dim))
            ext_layers.append(nn.LeakyReLU())
            f_dim = d_model

        elif self.mode == 'image':  # image only
            self.autoencoder = AutoEncoder()

            while f_dim // 2 > d_model:
                ext_layers.append(nn.Linear(f_dim, f_dim // 2))
                ext_layers.append(nn.LayerNorm(f_dim // 2))
                ext_layers.append(nn.LeakyReLU())
                f_dim = f_dim // 2
            ext_layers.append(nn.Linear(f_dim, d_model - bio_embedding_dim))
            ext_layers.append(nn.LayerNorm(d_model - bio_embedding_dim))
            ext_layers.append(nn.LeakyReLU())
            f_dim = d_model

        else:  # feature only
            self.autoencoder = None
            self.extractor = None

            f_dim = meta_feature_size
            while f_dim * 2 < d_model:
                ext_layers.append(nn.Linear(f_dim, f_dim * 2))
                ext_layers.append(nn.LayerNorm(f_dim * 2))
                ext_layers.append(nn.LeakyReLU())
                f_dim = f_dim * 2
            ext_layers.append(nn.Linear(f_dim, d_model - bio_embedding_dim))
            ext_layers.append(nn.LayerNorm(d_model - bio_embedding_dim))
            ext_layers.append(nn.LeakyReLU())
            f_dim = d_model

        # last feature extractor setup
        self.extractor = nn.Sequential(*ext_layers)
        while f_dim > 64:
            fc_layers.append(nn.Linear(f_dim, f_dim//2))
            fc_layers.append(nn.LayerNorm(f_dim//2))
            fc_layers.append(nn.ReLU())
            f_dim = f_dim // 2
        fc_layers.append(nn.Linear(f_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if m is self.fc:
                #     init.xavier_normal_(m.weight)
                # else:
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

    def forward(self, img, feature, bio):
        bio = bio.squeeze(1)

        age = bio[:, 0]
        gender = bio[:, 1]
        freq = bio[:, 2]
        gamer = bio[:, 3]
        platform = bio[:, 4:7].float()
        genre = bio[:, 7]

        age_emb = self.age_embedding(age.unsqueeze(1).float())
        gender_onehot = F.one_hot(gender, num_classes=self.gender_size).float()
        freq_onehot = F.one_hot(freq, num_classes=self.play_freq_size).float()
        gamer_onehot = F.one_hot(gamer, num_classes=self.gamer_size).float()
        genre_emb = self.genre_embedding(genre)

        bio_feature = torch.cat([age_emb, gender_onehot, freq_onehot, platform, gamer_onehot, genre_emb], dim=1)
        bio_ext = self.bio_extractor(bio_feature)
        # copy bio_ext to match the sequence length
        bio_ext = bio_ext.unsqueeze(1).repeat(1, self.window_size, 1)

        if self.mode == 'prefab' or self.mode == 'non_ordinal':
            reshaped_img = img.view(-1, *img.shape[2:])  # batch, win_size, c, h, w => batch * win_size, c, h, w
            e, d = self.autoencoder(reshaped_img)
            e = e.view(int(e.shape[0]/self.window_size), self.window_size, -1)

            e2 = self.extractor(e.view(-1, e.shape[-1]))
            e2 = e2.view(int(e2.shape[0]/self.window_size), self.window_size, -1)

            e3 = torch.cat((e2, feature, bio_ext), dim=-1)  # batch, sequence, feature
            ti = self.pos_encoder(e3)

            x = self.transformer_encoder(ti)
            avg_pooled = torch.mean(x, dim=1)
            x = self.fc(avg_pooled)

        elif self.mode == 'image':
            reshaped_img = img.view(-1, *img.shape[2:])  # batch, win_size, c, h, w => batch * win_size, c, h, w
            e, d = self.autoencoder(reshaped_img)
            e = e.view(int(e.shape[0] / self.window_size), self.window_size, -1)

            e2 = self.extractor(e.view(-1, e.shape[-1]))
            e2 = e2.view(int(e2.shape[0] / self.window_size), self.window_size, -1)

            e3 = torch.cat((e2, bio_ext), dim=-1)  # batch, sequence, feature
            ti = self.pos_encoder(e3)

            x = self.transformer_encoder(ti)
            avg_pooled = torch.mean(x, dim=1)
            x = self.fc(avg_pooled)

        else:
            e = self.extractor(feature)
            e2 = torch.cat((e, bio_ext), dim=-1)  # batch, sequence, feature
            ti = self.pos_encoder(e2)

            x = self.transformer_encoder(ti)
            avg_pooled = torch.mean(x, dim=1)
            x = self.fc(avg_pooled)
            d = None

        return x, d



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # d_model must be even number
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
