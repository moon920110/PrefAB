import os

import pandas as pd
from tqdm import tqdm

from utils.stats import get_dtw_cluster


class CustomReader:
    def __init__(self, data, config=None, logger=None):
        self.config = config
        self.logger = logger

        self.data = data
        self.bio = pd.read_csv(os.path.join(config['data']['path'], 'raw_data', 'biographical_data_with_genre.csv'),
                               encoding='utf-8',
                               low_memory=False)
        self.player = config['experiment']['player']
        self.session = config['experiment']['session']
        self.game_name = config['game_name'][config['experiment']['game']]

    def prepare_dataset(self):

        self.bio['Genre_idx'] = pd.factorize(self.bio['Genre'])[0]
        gender_size = self.bio['Gender'].unique().size
        play_freq_size = self.bio['Play Frequency'].unique().size
        gamer_size = self.bio['Gamer'].unique().size
        genre_size = self.bio['Genre_idx'].unique().size
        bio_size = {'gender': gender_size, 'play_freq': play_freq_size, 'gamer': gamer_size, 'genre': genre_size}

        self.data.sort_values('time_index')
        self.data['arousal'] = 0
        self.data['cluster'] = 0
        self.data['player_idx'] = 0
        self.data['pair_rank_label'] = 0
        self.data['game_idx'] = 0
        self.data['arousal_window_mean'] = 0

        player_bio = self.bio[self.bio['ParticipantID'] == self.player]
        player_bio = player_bio.drop(columns=['ParticipantID', 'Metacritic Code', 'Genre'])

        video_name = f'{self.player}_{self.game_name}_{self.session}.h5'
        video_full_path = os.path.join(self.config['data']['path'], self.config['data']['vision']['frame'], video_name)
        x = [[self.data, video_full_path, player_bio]]

        numeric_columns = self.data.select_dtypes(include=['number']).columns
        return x, numeric_columns, bio_size


class AgainReader:
    def __init__(self, config=None, logger=None, again_file_name: str=None):
        self.data_path = config['data']['path']
        self.config = config
        self.logger = logger

        if again_file_name is None:
            again_file_name = 'clean_data.csv'
        bio_file_name = 'biographical_data_with_genre.csv'

        if self.logger is not None:
            self.logger.info(f'data from {os.path.join(self.data_path, "clean_data", again_file_name)}')
            self.logger.info(f'bio data from {os.path.join(self.data_path, "raw_data", bio_file_name)}')
        # read csv without row index
        self.again = pd.read_csv(os.path.join(self.data_path, 'clean_data', again_file_name),
                                 encoding='utf-8',
                                 low_memory=False)
        self.bio = pd.read_csv(os.path.join(self.data_path, 'raw_data', bio_file_name),
                               encoding='utf-8',
                               low_memory=False)
        self.again.columns = [col.split(']')[-1] for col in self.again.columns]

    def _prepare_ordinal_dataset(self, domain, scope):
        if domain == 'game':
            again = self.game_info_by_name(scope)
            again['game_idx'] = 0
        elif domain == 'genre':
            if scope != "all":
                again = self.game_info_by_genre(scope)
            else:
                again = self.again
            games = again['game'].unique()
            for game in games:
                again.loc[again['game'] == game, 'game_idx'] = self.config['game_numbering'][scope][game] / self.config['game_numbering']['game_cnt_per_genre']
        else:
            again = self.again
        again['player_idx'] = pd.factorize(again['player_id'])[0]

        # get mean arousal for each 12 frames
        again['arousal_window_mean'] = again.groupby(['session_id'])['arousal'].transform(lambda x: x.rolling(self.config['train']['window_size'], 1).mean())

        # get arousal diff by 4 frame interval
        arousal_diff = again.groupby(['session_id'])['arousal_window_mean'].diff(self.config['train']['window_stride'])
        # NaN and 0 to 1 negative to 0 positive to 2
        pair_rank_label = arousal_diff.apply(lambda x: 1 if pd.isna(x) or x == 0 else 0 if x < 0 else 2)

        again['pair_rank_label'] = pair_rank_label
        if self.config['clustering']['activate']:
            cluster_data_path = os.path.join(self.data_path, 'cluster', f'cluster_{scope}.csv')
            if not os.path.exists(cluster_data_path):
                print("compute new cluster")
                clusters = get_dtw_cluster(again, self.config)
                if self.config['clustering']['caching']:
                    if not os.path.exists(os.path.join(self.data_path, 'cluster')):
                        os.mkdir(os.path.join(self.data_path, 'cluster'))
                    clusters = pd.DataFrame(clusters.items(), columns=['session_id', 'cluster'])
                    clusters.to_csv(cluster_data_path, index=False)
            else:
                clusters = pd.read_csv(cluster_data_path)
            again['cluster'] = again['session_id'].map(clusters.set_index('session_id')['cluster'])
            if self.config['clustering']['cluster_sample'] != 0:
                cluster_idx = self.config['clustering']['cluster_sample'] - 1
                again = again[again['cluster'] == cluster_idx]

        return again

    def prepare_sequential_ranknet_dataset(self):
        domain = self.config['train']['domain']
        scope = self.config['train']['genre'] if domain == 'genre' else self.config['train']['game']
        data = self._prepare_ordinal_dataset(domain=domain, scope=scope)
        x = []
        game_metadata = []
        total_iter = len(data['session_id'].unique())
        pbar = tqdm(total=total_iter, desc='Processing AGAIN dataset to sequential data')
        numeric_columns = data.select_dtypes(include=['number']).columns

        self.bio['Genre_idx'] = pd.factorize(self.bio['Genre'])[0]
        gender_size = self.bio['Gender'].unique().size
        play_freq_size = self.bio['Play Frequency'].unique().size
        gamer_size = self.bio['Gamer'].unique().size
        genre_size = self.bio['Genre_idx'].unique().size
        bio_size = {'gender': gender_size, 'play_freq': play_freq_size, 'gamer': gamer_size, 'genre': genre_size}

        # a game log for each player: one session
        for session in data['session_id'].unique():
            pbar.update(1)
            player_data = data[(data['session_id'] == session)]
            player = player_data['player_id'].unique()[0]
            game = player_data['game'].unique()[0]
            if len(player_data) == 0:
                continue
            video_name = f'{player}_{self.config["game_name"][game]}_{session}'
            # video_full_path = os.path.join(self.data_path, self.config['data']['vision']['frame'], video_name)

            player_data = player_data.sort_values('time_index')

            player_bio = self.bio[self.bio['ParticipantID'] == player]
            player_bio = player_bio.drop(columns=['ParticipantID', 'Metacritic Code', 'Genre'])
            if player_bio.empty:
                continue

            x.append([player_data, video_name, player_bio])
            game_metadata.append(game)

        return x, numeric_columns, bio_size, game_metadata

    def game_info_by_genre(self, genre):
        again = self.again[self.again['genre'] == genre].dropna(axis=1, how='any')
        again = again.loc[:, (again != 0).any(axis=0)]
        return again

    def game_info_by_name(self, game_name):
        again = self.again[self.again['game'] == game_name].dropna(axis=1, how='any')
        again = again.loc[:, (again != 0).any(axis=0)]
        return again

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

    def compute_second_order_diff(self):
        """Compute second order difference of arousal to find non-linear point"""
        self.again['arousal_diff'] = self.again.groupby(['player_id', 'game'])['arousal'].diff()
        self.again['arousal_diff_2'] = self.again.groupby(['player_id', 'game'])['arousal_diff'].diff()


if __name__ == "__main__":
    again_reader = AgainReader()
    # logging.debug(again_reader.load_ranknet_train_eval())
