import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from network.autoencoder import AutoEncoder
from network.film import FiLM


class Prefab(nn.Module):
    def __init__(self, config, meta_feature_size, bio_feature_size):
        super(Prefab, self).__init__()
        self.config = config
        self.window_size = config['train']['window_size']
        self.mode = config['train']['mode']
        ext_layers = []
        main_fc_layers = []
        aux_fc_layers = []
        d_model = config['train']['d_model']
        self.genre_size = bio_feature_size['genre']
        self.gender_size = bio_feature_size['gender']
        self.play_freq_size = bio_feature_size['play_freq']
        self.gamer_size = bio_feature_size['gamer']

        bio_embedding_dim = config['train']['bio_embedding_dim']

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
        self.FiLM = FiLM(bio_total_size, d_model)

        # transformer setup
        self.pos_encoder = PositionalEncoding(d_model, dropout=config['train']['dropout'])
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=config['train']['dropout'], batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.config['train']['num_transform_layers'])

        # autoencoder and feature extractor setup
        if self.mode == 'prefab' or self.mode == 'non_ordinal':
            self.autoencoder = AutoEncoder()
            f_dim = self._compute_f_dim()

            while f_dim // 2 > d_model:
                ext_layers.append(nn.Linear(f_dim, f_dim//2))
                ext_layers.append(nn.LayerNorm(f_dim//2))
                ext_layers.append(nn.LeakyReLU())
                f_dim = f_dim // 2
            ext_layers.append(nn.Linear(f_dim, d_model - meta_feature_size))
            ext_layers.append(nn.LayerNorm(d_model - meta_feature_size))
            ext_layers.append(nn.LeakyReLU())
            f_dim = d_model

        elif self.mode == 'image':  # image only
            self.autoencoder = AutoEncoder()
            f_dim = self._compute_f_dim()

            while f_dim // 2 > d_model:
                ext_layers.append(nn.Linear(f_dim, f_dim // 2))
                ext_layers.append(nn.LayerNorm(f_dim // 2))
                ext_layers.append(nn.LeakyReLU())
                f_dim = f_dim // 2
            ext_layers.append(nn.Linear(f_dim, d_model))
            ext_layers.append(nn.LayerNorm(d_model))
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
            ext_layers.append(nn.Linear(f_dim, d_model))
            ext_layers.append(nn.LayerNorm(d_model))
            ext_layers.append(nn.LeakyReLU())
            f_dim = d_model

        # last feature extractor setup
        self.extractor = nn.Sequential(*ext_layers)
        while f_dim > 64:
            main_fc_layers.append(nn.Linear(f_dim, f_dim//2))
            main_fc_layers.append(nn.LayerNorm(f_dim//2))
            main_fc_layers.append(nn.ReLU())

            aux_fc_layers.append(nn.Linear(f_dim, f_dim//2))
            aux_fc_layers.append(nn.LayerNorm(f_dim//2))
            aux_fc_layers.append(nn.ReLU())

            f_dim = f_dim // 2
        main_fc_layers.append(nn.Linear(f_dim, 1))
        aux_fc_layers.append(nn.Linear(f_dim, config['clustering']['n_clusters']))
        self.main_fc = nn.Sequential(*main_fc_layers)
        self.aux_fc = nn.Sequential(*aux_fc_layers)

        self._init_weights()

    def _init_weights(self):
        if self.config['train']['ablation']['film']:
            for param in self.FiLM.parameters():
                param.requires_grad = False
        if self.config['train']['ablation']['aux']:
            for param in self.aux_fc.parameters():
                param.requires_grad = False

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

    def _compute_f_dim(self):
        dummy_input = torch.randn(1, 3, *self.config['data']['transform_size'])  # Adjust as needed
        with torch.no_grad():
            latent_vector, _  = self.autoencoder(dummy_input)
        return latent_vector.view(1, -1).shape[1]  # Flatten and get size

    def forward(self, img, feature, bio, test=False):
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
        if self.config['train']['ablation']['film']:
            film_gamma = torch.ones(bio_feature.shape[0], 1).to(bio_feature.device)
            film_beta = torch.zeros(bio_feature.shape[0], 1).to(bio_feature.device)
        else:
            film_gamma, film_beta = self.FiLM(bio_feature)
        # copy bio_ext to match the sequence length
        # bio_ext = bio_ext.unsqueeze(1).repeat(1, self.window_size, 1)

        if self.mode == 'prefab' or self.mode == 'non_ordinal':
            reshaped_img = img.view(-1, *img.shape[2:])  # batch, win_size, c, h, w => batch * win_size, c, h, w
            e, d = self.autoencoder(reshaped_img)
            e = e.view(int(e.shape[0]/self.window_size), self.window_size, -1)

            e2 = self.extractor(e.view(-1, e.shape[-1]))
            e2 = e2.view(int(e2.shape[0]/self.window_size), self.window_size, -1)

            f_z = torch.cat((e2, feature), dim=-1)  # batch, sequence, feature
            ti = self.pos_encoder(f_z)

            x = self.transformer_encoder(ti)
            avg_pooled = torch.mean(x, dim=1)

        elif self.mode == 'image':
            reshaped_img = img.view(-1, *img.shape[2:])  # batch, win_size, c, h, w => batch * win_size, c, h, w
            e, d = self.autoencoder(reshaped_img)
            e = e.view(int(e.shape[0] / self.window_size), self.window_size, -1)

            e2 = self.extractor(e.view(-1, e.shape[-1]))
            f_z = e2.view(int(e2.shape[0] / self.window_size), self.window_size, -1)

            ti = self.pos_encoder(f_z)

            x = self.transformer_encoder(ti)
            avg_pooled = torch.mean(x, dim=1)

        else:
            f_z = self.extractor(feature)
            ti = self.pos_encoder(f_z)

            x = self.transformer_encoder(ti)
            avg_pooled = torch.mean(x, dim=1)
            d = None
        z = avg_pooled * film_gamma + film_beta
        x = self.main_fc(z)
        if self.config['train']['ablation']['aux']:
            a = self.aux_fc(z.detach())
        else:
            a = self.aux_fc(z)

        if test:
            return x, a, d, z
        return x, a, d



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
