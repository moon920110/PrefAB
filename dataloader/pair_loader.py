import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from dataloader.again_reader import AgainReader


class PairLoader(Dataset):
    def __init__(self, config, logger=None):
        self.dataset, self.numeric_columns = AgainReader(config, logger).prepare_sequential_ranknet_dataset()
        self.config = config

        self.x_img_pairs = []
        self.x_meta_pairs = []
        self.y = []

        self.transform = transforms.Compose([
            transforms.Resize(self.config['data']['transform_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['data']['img_mean'], std=self.config['data']['img_std'])
        ])

        self.sequential = self.config['data']['sequential']
        if self.sequential:
            self.init_sequence_dataset()
        else:
            self.init_dataset()

    def init_sequence_dataset(self):
        for x, img_path in tqdm(self.dataset, desc=f'Preparing sequential dataset'):
            for game in x['game'].unique():
                for player in x['player_id'].unique():
                    player_data = x[(x['game'] == game) & (x['player_id'] == player)]
                    if len(player_data) == 0:
                        continue

                    for idx in range(5, len(player_data)):
                        seq = player_data.iloc[idx-5:idx]  # stack 5 frames (0~3), (1~4) pair
                        img_data = [img_path, seq['time_index'].values, seq['time_stamp'].values]
                        y = seq['pair_rank_label'].values[-1]  # label of the last frame
                        seq.loc[:, 'game'] = self.config['game_numbering'][self.config['train']['genre']][game]
                        seq = seq.loc[:, self.numeric_columns]
                        seq = seq.drop(columns=['pair_rank_label']).values.astype(np.float32)

                        self.x_img_pairs.append(img_data)
                        self.x_meta_pairs.append(seq)
                        self.y.append(y)

    def init_dataset(self):
        for x, img_path in tqdm(self.dataset, desc=f'Preparing dataset'):
            for game in x['game'].unique():
                for player in x['player_id'].unique():
                    player_data = x[(x['game'] == game) & (x['player_id'] == player)]
                    if len(player_data) == 0:
                        continue
                    for _, row in player_data.iterrows():
                        img_data = [img_path, row['time_index'], row['time_stamp']]
                        y = row['pair_rank_label']
                        row['game'] = self.config['game_numbering'][self.config['train']['genre']][game]
                        row = row.loc[self.numeric_columns]
                        row = row.drop(columns=['pair_rank_label']).astype(np.float32)

                        self.x_img_pairs.append(img_data)
                        self.x_meta_pairs.append(row)
                        self.y.append(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_data = self.x_img_pairs[idx]  # images: (0:4) = comparison frames, (1:5) = main frames
        meta = self.x_meta_pairs[idx]  # game log data
        y = self.y[idx]  # 0, 0.5, 1

        img_path, time_indices, time_stamps = img_data
        if self.sequential:  # ranknet using sequential input
            with h5py.File(img_path, 'r') as f:
                # (sequence, height, width, channel)
                compare_frames = torch.stack(
                    [self.transform(Image.fromarray(np.array(f[f'frames/{time_index}_{time_stamp}'])))
                     for time_index, time_stamp in zip(time_indices[:4], time_stamps[:4])])
                # (sequence, height, width, channel) to (sequence, channel, height, width) // open cv BGR format 0~255
                main_frames = torch.stack(
                    [self.transform(Image.fromarray(np.array(f[f'frames/{time_index}_{time_stamp}'])))
                     for time_index, time_stamp in zip(time_indices[1:], time_stamps[1:])])

            # 4, 3, 320, 480 [frames]
            return compare_frames, \
                torch.tensor(meta[:4]), \
                main_frames, \
                torch.tensor(meta[1:]), \
                torch.tensor(y)

        else:
            with h5py.File(img_path, 'r') as f:
                frames = self.transform(Image.fromarray(np.array(f[f'frames/{time_indices}_{time_stamps}'])))
            return frames, torch.tensor(meta), torch.tensor(y)
