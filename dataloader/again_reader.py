import os
import pandas as pd


class AgainReader:
    def __init__(self, data_path='/home/jovyan/supermoon/AGAIN'):
        data_path = data_path

        self.again = pd.read_csv(os.path.join(data_path, 'clean_data', 'clean_data.csv'), encoding='utf-8', low_memory=False)

    def game_info_by_genre(self, genre):
        return self.again[self.again['[control]genre'] == genre].dropna(axis=1, how='any')

    def game_info_by_name(self, game_name):
        return self.again[self.again['[control]game'] == game_name].dropna(axis=1, how='any')

    def common_features(self):
        shooter_games = self.game_info_by_genre('Shooter')
        platform_games = self.game_info_by_genre('Platformer')
        racing_games = self.game_info_by_genre('Racing')

        return list(set(shooter_games.columns) & set(platform_games.columns) & set(racing_games.columns))

    def unique_game_info(self):
        return self.again.drop_duplicates(subset='[control]game', keep='first')

    def available_feature_names_by_game(self, game_name):
        game_info = self.unique_game_info()
        return game_info[game_info['[control]game'] == game_name].dropna(axis=1, how='any').columns


if __name__ == "__main__":
    again_reader = AgainReader()
    print(again_reader.available_feature_names_by_game('Heist!'))
