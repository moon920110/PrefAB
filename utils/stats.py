import pandas as pd
import yaml
import dtw
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score


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


def get_dtw_cluster(data, config):
    n_cluster = config['clustering']['n_clusters']
    metric = config['clustering']['metric']
    metric_params = config['clustering']['metric_params']
    random_state = config['clustering']['random_state']
    verbose = config['clustering']['verbose']
    input_type = config['clustering']['input_type']

    sessions = data['session_id'].unique()
    # get max length of arousal group by session
    min_length = 99999999
    for session in sessions:
        item_len = len(data[data['session_id'] == session])
        min_length = min(item_len, min_length)

    session_data = []
    for session in sessions:
        arousal = data[data['session_id'] == session][input_type].values
        if len(arousal) > min_length:
            arousal = arousal[:min_length]
        session_data.append(arousal)

    session_data = np.array(session_data)

    kmeans = TimeSeriesKMeans(n_clusters=n_cluster, metric=metric, verbose=verbose, random_state=random_state, metric_params=metric_params)
    p = kmeans.fit_predict(session_data)
    silhouette_avg = silhouette_score(session_data, p, metric=metric, metric_params=metric_params)
    print(f'silhouette score: {silhouette_avg}')

    session_cluster = {}
    for session, cluster in zip(sessions, p):
        session_cluster[session] = cluster

    if config['clustering']['visualize']:
        fig, axs = plt.subplots(1, n_cluster)
        for yi, ax in enumerate(axs):
            for xx in session_data[p == yi]:
                ax.plot(xx.ravel(), "k-", alpha=0.3)
            ax.plot(kmeans.cluster_centers_[yi].ravel(), "r-")
            ax.text(0.0, 0.0, f'Cluster ({yi + 1}): {len(session_data[p == yi])}', transform=ax.transAxes)
        plt.show()

    return session_cluster


if __name__ == "__main__":
    from dataloader.again_reader import AgainReader
    with open('../config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    game = 'Shootout'
    print(f'read data {game}')
    again_reader = AgainReader(config)
    # data = again_reader.game_info_by_name(game)
    data = again_reader.game_info_by_name('Shootout')
    print(f'len data: {len(data)}')

    get_dtw_cluster(data, config)

    #######################################################################
    # feature_names = again_reader.available_feature_names_by_game(game)
    # print(f'available feature names: {feature_names}')
    # print('calc total correlation')
    # total_corr = calc_correlation(data)
    # print('calc correlation per player')
    # corr_dict = calc_correlation_per_player(data)
    # # corr_dict = {'total': total_corr, **corr_dict}
    #
    # print(f'save to ../data/{game}_correlation.csv')
    # corr_dict_df = pd.DataFrame(corr_dict)
    # mean = corr_dict_df.mean(axis=1)
    # std = corr_dict_df.std(axis=1)
    #
    # corr_dict_df['mean'] = mean
    # corr_dict_df['std'] = std
    # corr_dict_df['total'] = total_corr
    #
    # # corr_dict_df.T.dropna(axis=1, how='all').to_csv(f'../data/{game}_correlation.csv')
    # corr_dict_df.T.to_csv(f'../data/{game}_correlation.csv')
