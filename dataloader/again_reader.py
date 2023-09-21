import os

import pandas as pd
from tqdm import tqdm


class AgainReader:
    def __init__(self, config=None, logger=None):
        self.data_path = config['data']['path']
        self.config = config
        self.logger = logger

        self.logger.debug(f'data from {os.path.join(self.data_path, "clean_data", "clean_data.csv")}')
        # read csv without row index
        self.again = pd.read_csv(os.path.join(self.data_path, 'clean_data', 'clean_data.csv'),
                                 encoding='utf-8',
                                 low_memory=False)
        self.again.columns = [col.split(']')[-1] for col in self.again.columns]

    def _prepare_ordinal_dataset(self, target, target_name):
        if target == 'game':
            again = self.game_info_by_name(target_name)
        elif target == 'genre':
            again = self.game_info_by_genre(target_name)
        else:
            again = self.again

        # get arousal diff by frame interval
        arousal_diff = again.groupby(['player_id', 'game'])['arousal'].diff()
        # NaN and 0 to 0.5 negative to 0 positive to 1
        pair_rank_label = arousal_diff.apply(lambda x: 0.5 if pd.isna(x) or x == 0 else 0 if x < 0 else 1)

        again['pair_rank_label'] = pair_rank_label
        again = again.drop(columns=['arousal'])

        return again

    def prepare_sequential_ranknet_dataset(self):
        data = self._prepare_ordinal_dataset(target='genre', target_name=self.config['train']['genre'])
        x = []
        total_iter = len(data['game'].unique()) * len(data['player_id'].unique())
        pbar = tqdm(total=total_iter, desc='Preparing sequential dataset')
        numeric_columns = data.select_dtypes(include=['number']).columns
        # add game at the first
        numeric_columns = ['game'] + list(numeric_columns)

        for game in data['game'].unique():
            for player in data['player_id'].unique():
                pbar.update(1)
                player_data = data[(data['game'] == game) & (data['player_id'] == player)]
                if len(player_data) == 0:
                    continue
                video_name = f'{player}_{self.config["game_name"][game]}_{player_data["session_id"].unique()[0]}.h5'
                video_full_path = os.path.join(self.data_path, self.config['data']['vision']['frame'], video_name)

                player_data = player_data.sort_values('time_index')
                pads = pd.concat([player_data.iloc[0]] * 3, axis=1).T
                player_data = pd.concat([pads, player_data], ignore_index=True)

                x.append([player_data, video_full_path])

        return x, numeric_columns

    def game_info_by_genre(self, genre):
        return self.again[self.again['genre'] == genre].dropna(axis=1, how='any')

    def game_info_by_name(self, game_name):
        return self.again[self.again['game'] == game_name].dropna(axis=1, how='any')

    def common_features(self):
        shooter_games = self.game_info_by_genre('Shooter')
        platform_games = self.game_info_by_genre('Platformer')
        racing_games = self.game_info_by_genre('Racing')

        return list(set(shooter_games.columns) & set(platform_games.columns) & set(racing_games.columns))

    def unique_game_info(self):
        return self.again.drop_duplicates(subset='game', keep='first')

    def available_feature_names(self, target, target_name):
        game_info = self.unique_game_info()
        return game_info[game_info[target] == target_name].dropna(axis=1, how='any').columns


if __name__ == "__main__":
    again_reader = AgainReader()
    # logging.debug(again_reader.load_ranknet_train_eval())
