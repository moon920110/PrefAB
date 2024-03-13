import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from dataloader.again_reader import AgainReader


class PairDataset(Dataset):
    def __init__(self, config, logger=None):
        self.dataset, self.numeric_columns = AgainReader(config, logger).prepare_sequential_ranknet_dataset()
        self.config = config
        self.window_size = config['train']['window_size']
        self.window_stride = config['train']['window_stride']

        self.x_img_pairs = []
        self.x_meta_pairs = []
        self.y = []

        self.transform = transforms.Compose([
            transforms.Resize(self.config['data']['transform_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['data']['img_mean'], std=self.config['data']['img_std'])
        ])

        self.init_sequence_dataset()

    def init_sequence_dataset(self):
        if self.config['debug']['activate']:
            self.dataset = self.dataset[:self.config['debug']['data_limit']]  # limit for a part of one game

        offset = self.window_size + self.window_stride
        for player_data, img_path in tqdm(self.dataset, desc=f'Preparing sequential dataset'):  # for each game and player
            if len(player_data) == 0:
                continue

            if self.config['debug']['activate'] and self.config['debug']['limit_player']:
                if player_data['player_idx'].unique()[0] != 0:
                    continue

            for idx in range(0, len(player_data)-offset+1):
                seq = player_data.iloc[idx:idx+offset]  # stack `window_size` frames (0~win_size-1), (4~win_size+win_stride) pair
                img_data = [img_path, seq['time_index'].values, seq['time_stamp'].values]
                y = seq['pair_rank_label'].values[-1]  # label of the last frame
                seq = seq.loc[:, self.numeric_columns]
                seq = seq.drop(columns=['player_idx', 'pair_rank_label', 'epoch', 'engine_tick', 'time_stamp', 'activity', 'score', 'game_idx', 'arousal', 'arousal_window_mean']).values.astype(np.float32)
                # TODO: Add player related information (biography)

                self.x_img_pairs.append(img_data)
                self.x_meta_pairs.append(seq)
                self.y.append(y)

    def __len__(self):
        return len(self.y)

    def compute_sample_weight(self, indices):
        tmp_y = np.array(self.y)[indices]
        _, cnts = np.unique(tmp_y, return_counts=True)
        class_sample_cnt = np.array(cnts)  # count of each label
        weight = class_sample_cnt.max() / class_sample_cnt

        sample_weight = np.array(weight[tmp_y.astype(int)])

        return sample_weight

    def get_meta_feature_size(self):
        return self.x_meta_pairs[0].shape[-1]

    def __getitem__(self, idx):
        img_data = self.x_img_pairs[idx]  # images: (0:4) = comparison frames, (1:5) = main frames
        meta = self.x_meta_pairs[idx]  # game log data
        y = self.y[idx]  # 0, 0.5, 1 => 0, 1, 2

        img_path, time_indices, time_stamps = img_data
        with h5py.File(img_path, 'r') as f:
            # (sequence, height, width, channel)
            compare_frames = torch.stack(
                [self.transform(Image.fromarray(np.array(f[f'frames/{time_index}_{time_stamp}'])))
                 for time_index, time_stamp in zip(time_indices[:self.window_size], time_stamps[:self.window_size])])
            # (sequence, height, width, channel) to (sequence, channel, height, width) // open cv BGR format 0~255
            main_frames = torch.stack(
                [self.transform(Image.fromarray(np.array(f[f'frames/{time_index}_{time_stamp}'])))
                 for time_index, time_stamp in zip(time_indices[-self.window_size:], time_stamps[-self.window_size:])])

        # 4, 3, 320, 480 [frames]
        return compare_frames, \
            torch.tensor(meta[:self.window_size]), \
            main_frames, \
            torch.tensor(meta[-self.window_size:]), \
            torch.tensor(y)


class TestDataset:
    def __init__(self, dataset, numeric_columns, config):
        self.dataset = dataset
        self.numeric_columns = numeric_columns
        self.config = config
        self.window_size = config['train']['window_size']

        self.x_img = []
        self.x_meta = []
        self.y = []
        self.player_idx = []

        self.transform = transforms.Compose([
            transforms.Resize(self.config['data']['transform_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['data']['img_mean'], std=self.config['data']['img_std'])
        ])

        self.init_sequence_dataset()

    def init_sequence_dataset(self):
        for player_data, img_path in tqdm(self.dataset, desc=f'Preparing sequential dataset for validation'):  # for each game and player
            if len(player_data) == 0:
                continue

            self.player_idx.append(len(self.y))
            for idx in range(0, len(player_data) - self.window_size):
                seq = player_data.iloc[idx:idx + self.window_size]  # stack `window_size` frames (0~win_size-1), (4~win_size+win_stride) pair
                img_data = [img_path, seq['time_index'].values, seq['time_stamp'].values]
                relative_y = seq['pair_rank_label'].values[-1]
                y = seq['arousal'].values[-1]
                mean_y = seq['arousal_window_mean'].values[-1]
                seq = seq.loc[:, self.numeric_columns]
                seq = seq.drop(
                    columns=['player_idx', 'pair_rank_label', 'epoch', 'engine_tick', 'time_stamp', 'activity', 'score',
                             'game_idx', 'arousal', 'arousal_window_mean']).values.astype(np.float32)

                self.x_img.append(img_data)
                self.x_meta.append(seq)
                self.y.append([y, relative_y, mean_y])

    def sample_player_data(self, size=10):
        p_indices = np.random.choice(len(self.player_idx)-1, size)
        return p_indices

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_data = self.x_img[idx]  # images per player
        meta = self.x_meta[idx]  # game log data per player
        y, r_y, m_y = self.y[idx]  # [arousal, relative_arousal, mean_arousal]

        img_path, time_indices, time_stamps = img_data
        with h5py.File(img_path, 'r') as f:
            # (sequence, height, width, channel)
            frames = torch.stack(
                [self.transform(Image.fromarray(np.array(f[f'frames/{time_index}_{time_stamp}'])))
                 for time_index, time_stamp in zip(time_indices[:self.window_size], time_stamps[:self.window_size])])
            # (sequence, height, width, channel) to (sequence, channel, height, width) // open cv BGR format 0~255

        # 4, 3, 320, 480 [frames]
        return frames, \
            torch.tensor(meta[:self.window_size]), \
            torch.tensor(y), \
            torch.tensor(r_y), \
            torch.tensor(m_y)
