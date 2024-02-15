import pandas as pd
import dtw

from dataloader.again_reader import AgainReader


def calc_correlation_per_player(data):
    out = 'arousal_delta'

    players = data['player_id'].unique()

    corr_dict = {}
    for player in players:
        player_data = data[data['player_id'] == player]
        player_data.loc[:, out] = player_data['arousal'].diff()
        numerics = player_data.select_dtypes(include='number')
        corr = numerics.corr()[out]
        corr_dict[player] = corr

    return corr_dict


def calc_correlation(data):
    out = 'arousal_delta'

    data.loc[:, out] = data['arousal'].diff()
    numerics = data.select_dtypes(include='number')
    corr = numerics.corr()[out]

    return corr


# TODO: calc dtw, clustering by dtw
def calc_dtw(data):
    players = data['player_id'].unique()
    games = data['game'].unique()

    for player in players:
        for game in games:
            player_data = data[(data['player_id'] == player) & (data['game'] == game)]
            player_data = player_data['arousal'].values
            dist, cost, acc, path = dtw.dtw(player_data[:, 0], player_data[:, 1], dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            print(f'{player} - {game}: {dist}')

if __name__ == "__main__":
    game = 'Shootout'
    print(f'read data {game}')
    again_reader = AgainReader()
    # data = again_reader.game_info_by_name(game)
    data = again_reader.game_info_by_genre('Shooter')
    print(f'len data: {len(data)}')
    feature_names = again_reader.available_feature_names_by_game(game)
    print(f'available feature names: {feature_names}')
    print('calc total correlation')
    total_corr = calc_correlation(data)
    print('calc correlation per player')
    corr_dict = calc_correlation_per_player(data)
    # corr_dict = {'total': total_corr, **corr_dict}

    print(f'save to ../data/{game}_correlation.csv')
    corr_dict_df = pd.DataFrame(corr_dict)
    mean = corr_dict_df.mean(axis=1)
    std = corr_dict_df.std(axis=1)

    corr_dict_df['mean'] = mean
    corr_dict_df['std'] = std
    corr_dict_df['total'] = total_corr

    # corr_dict_df.T.dropna(axis=1, how='all').to_csv(f'../data/{game}_correlation.csv')
    corr_dict_df.T.to_csv(f'../data/{game}_correlation.csv')
