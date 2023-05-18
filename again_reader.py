import os
import numpy as np
import pandas as pd


data_path = '/Users/supermoon/Documents/Research/AGAIN dataset/AGAIN'

again = pd.read_csv(os.path.join(data_path, 'clean_data', 'clean_data.csv'), encoding='utf-8')

# get data where genre is Shooter without null columns
shooter_games = again[again['[control]genre'] == 'Shooter'].dropna(axis=1, how='any')
platform_games = again[again['[control]genre'] == 'Platformer'].dropna(axis=1, how='any')
racing_games = again[again['[control]genre'] == 'Racing'].dropna(axis=1, how='any')

# 위에 세 개 데이터셋에서 공통적으로 나오는 column만 추출
common_columns = list(set(shooter_games.columns) & set(platform_games.columns) & set(racing_games.columns))