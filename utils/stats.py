import pandas as pd

from dataloader.again_reader import AgainReader


def calc_correlation_per_player(data):
    out = '[output]arousal'

    players = data['[control]player_id'].unique()

    corr_dict = {}
    for player in players:
        player_data = data[data['[control]player_id'] == player]
        player_data.loc[:, 'arousal_delta'] = player_data['[output]arousal'].diff()
        numerics = player_data.select_dtypes(include='number')
        corr = numerics.corr()[out]
        corr_dict[player] = corr

    return corr_dict


def calc_correlation(data):
    out = '[output]arousal'

    data.loc[:, 'arousal_delta'] = data['[output]arousal'].diff()
    print(data)
    numerics = data.select_dtypes(include='number')
    corr = numerics.corr()[out]

    return corr


if __name__ == "__main__":
    game = 'Heist!'
    print(f'read data {game}')
    again_reader = AgainReader()
    data = again_reader.game_info_by_name(game)
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

    corr_dict_df.T.to_csv(f'../data/{game}_correlation.csv')
