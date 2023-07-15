from dataloader.again_reader import AgainReader
from utils.vis import *


if __name__ == '__main__':
    again_reader = AgainReader(data_path='./')
    title = 'Shooter'
    data = again_reader.game_info_by_genre(title)
    # find player id
    plot_ordinal_arousal(data, title)
