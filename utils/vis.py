import matplotlib.pyplot as plt

from dataloader.again_reader import AgainReader


def plot_ordinal_arousal(data, title):
    games = data['[control]game'].unique()

    fig_num = 1
    for game in games:
        game_data = data[data['[control]game'] == game].groupby('[control]player_id')

        plt.figure(fig_num)
        plt.title(f'{title} - {game}')
        for player_id, player_data in game_data:
            # compare arousal at a time point with the previous time point. If it is higher, it is 1, if it is lower, it is 0, and if it is the same, it is 0.5.
            ordinal_arousal = player_data['[output]arousal'].diff().apply(lambda x: 1 if x > 0 else 0 if x < 0 else 0.5)
            player_data['ordinal_arousal'] = ordinal_arousal

            plt.plot(player_data['[control]time_index'], player_data['ordinal_arousal'])
        fig_num += 1
    plt.show()


def plot_arousal(data, title):
    player_ids = data['[control]player_id'].unique()
    games = data['[control]game'].unique()
    # player_data = data[data['[control]player_id'] == player_id]

    fig_num = 1
    for game in games:
        game_data = data[data['[control]game'] == game].groupby('[control]player_id')
        # fig, ax = plt.subplots()
        # ax.title(f'{title}_{game}')
        plt.figure(fig_num)
        plt.title(f'{title}_{game}')
        i = 0
        for player_id, player_data in game_data:
            plt.plot(player_data['[control]time_index'], player_data['[output]arousal'])
            i += 1
            if i > 10:
                break
        fig_num += 1
    plt.show()


def plot_arousal_with_video(data, title):
    pass


if __name__ == '__main__':
    again_reader = AgainReader()
    title = 'Shooter'
    data = again_reader.game_info_by_genre(title)
    # find player id
    plot_arousal_with_video(data, title)
