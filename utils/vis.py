import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip
import cv2

from dataloader.again_reader import AgainReader


def plot_ordinal_arousal(data, title):
    games = data['[control]game'].unique()

    fig_num = 1
    for game in games:
        game_data = data[data['game'] == game].groupby('player_id')

        plt.figure(fig_num)
        plt.title(f'{title} - {game}')
        i = 0
        for player_id, player_data in game_data:
            # compare arousal at a time point with the previous time point. If it is higher, it is 1, if it is lower, it is 0, and if it is the same, it is 0.5.
            ordinal_arousal = player_data['arousal'].diff().apply(lambda x: 1 if x > 0 else 0 if x < 0 else 0.5)
            player_data['ordinal_arousal'] = ordinal_arousal

            plt.plot(player_data['time_index'], player_data['ordinal_arousal'])
            i += 1
            if i > 1:
                break
        fig_num += 1
        # plt.savefig(f'data/{title}-{game}.png')
    plt.show()


def plot_arousal(data, title):
    games = data['game'].unique()

    fig_num = 1
    for game in games:
        game_data = data[data['game'] == game].groupby('player_id')
        # fig, ax = plt.subplots()
        # ax.title(f'{title}_{game}')
        plt.figure(fig_num)
        plt.title(f'{title}_{game}')
        i = 0
        for player_id, player_data in game_data:
            plt.plot(player_data['time_index'], player_data['arousal'])
            i += 1
            if i > 10:
                break
        fig_num += 1
    plt.show()


if __name__ == "__main__":
    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    again = AgainReader(config).game_info_by_name('Shootout')
    again['time_index'] = again['time_index'].apply(
        lambda x: sum([a * b for a, b in zip([3600, 60, 1], map(float, x[7:].split(':')))]))
    player = again['player_id'].unique()[0]
    arousal_data = again[again['player_id'] == player]
    arousal_data = arousal_data.sort_values('time_index')

    plot_arousal(arousal_data, 'Shootout')