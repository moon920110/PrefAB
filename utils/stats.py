import os
import glob

import pandas as pd
import dtw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from scipy.signal import find_peaks

from utils.utils import read_scalar_summary


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


# TODO: peak를 구하는 근거 찾아서 정교화할 것, 현재 top peak 찾으면 탐지가 안 되는 애들이 좀 있음.
# + tensorboard에 기록된 데이터 가져와서 predict에서 peak 찾아서 ground truth와 비교해보기
def find_significant_peaks_and_valleys(
        data, threshold=0.1, prominence=None, distance=None, title=None, save_dir=None, show=False):
    smoothed_data = np.convolve(data, np.ones(10) / 10, mode='same')
    peaks, p_p = find_peaks(smoothed_data, prominence=prominence, distance=distance)
    valleys, p_v = find_peaks(-smoothed_data, prominence=prominence, distance=distance)

    peak_prominences = smoothed_data[peaks] - np.min(smoothed_data)
    valley_depths = np.max(smoothed_data) - smoothed_data[valleys]

    peak_threshold = int(len(peaks) * threshold)
    valley_threshold = int(len(valleys) * threshold)

    top_peaks = peaks[np.argsort(peak_prominences)[-peak_threshold:]]
    top_valleys = valleys[np.argsort(valley_depths)[-valley_threshold:]]

    # show peaks and valleys
    plt.plot(data)
    plt.plot(top_peaks, data[top_peaks], "x")
    plt.plot(top_valleys, data[top_valleys], "x")
    if title:
        plt.title(title)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{title}.png'))
            print(f'save to {os.path.join(save_dir, f"{title}.png")}')
    if show:
        plt.show()

    plt.close()

    return top_peaks, top_valleys


def post_analysis(root):
    log_dirs = glob.glob(os.path.join(root, 'test_epc[5-9][0-9]*'))
    log_dict = {}
    for log_dir in log_dirs:
        dir_name = log_dir.split('/')[-1]
        session = dir_name[:-8]
        tag = dir_name[-7:]

        if session not in log_dict:
            log_dict[session] = {}
        if tag == 'arousal':
            log_dict[session]['arousal'] = log_dir
        else:
            log_dict[session]['predict'] = log_dir

    font = FontProperties()
    font.set_weight('bold')
    font.set_size(14)

    accs = []
    errors = []
    for session, tags in log_dict.items():
        arousal_path = tags['arousal']
        predict_path = tags['predict']

        arousal_event_files = glob.glob(os.path.join(arousal_path, "events.out.tfevents.*"))
        predict_event_files = glob.glob(os.path.join(predict_path, "events.out.tfevents.*"))
        _, arousal_summary = read_scalar_summary(arousal_event_files[0])
        _, predict_summary = read_scalar_summary(predict_event_files[0])

        arousal_peaks, arousal_valleys = find_significant_peaks_and_valleys(arousal_summary, threshold=0, prominence=0.01, distance=20)
        predict_peaks, predict_valleys = find_significant_peaks_and_valleys(predict_summary, threshold=0, prominence=0.01, distance=20)

        correct_peak_cnt = 0
        error_peak_cnt = 0
        for arousal_peak in arousal_peaks:
            # find nearest peak in predict within +-10 distance
            nearest_peak = np.argmin(np.abs(arousal_peak - predict_peaks))
            if np.abs(arousal_peak - predict_peaks[nearest_peak]) < 10:  # half of distance
                correct_peak_cnt += 1
        accs.append(correct_peak_cnt / len(predict_peaks))

        for predict_peak in predict_peaks:
            # find nearest peak in predict within +-10 distance
            nearest_peak = np.argmin(np.abs(predict_peak - arousal_peaks))
            if np.abs(predict_peak - arousal_peaks[nearest_peak]) >= 10:
                error_peak_cnt += 1
        errors.append(error_peak_cnt / len(predict_peaks))

        plt.plot(arousal_summary, label='arousal (ground truth)')
        plt.plot(predict_summary, color='gray', linestyle='--', alpha=0.5, label='predicted arousal')
        plt.plot(arousal_peaks, arousal_summary[arousal_peaks], "r*", label='true peaks')
        plt.plot(arousal_valleys, arousal_summary[arousal_valleys], "r*")
        plt.plot(predict_peaks, arousal_summary[predict_peaks], "g*")
        plt.plot(predict_valleys, arousal_summary[predict_valleys], "g*")
        plt.plot(predict_peaks, predict_summary[predict_peaks], "g*", alpha=0.5, label='predicted peaks')
        plt.plot(predict_valleys, predict_summary[predict_valleys], "g*", alpha=0.5)

        for peak, valley in zip(predict_peaks, predict_valleys):
            plt.vlines(peak, arousal_summary[peak], predict_summary[peak], colors='green', linestyles=':', alpha=0.5)
            plt.vlines(valley, arousal_summary[valley], predict_summary[valley], colors='green', linestyles=':', alpha=0.5)

        plt.title(session)
        plt.legend()

        save_dir = os.path.join(root.split('/')[0], 'peak', root.split('/')[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # print(f'make dir: {save_dir}')

        plt.savefig(os.path.join(save_dir, f'{session}.png'))
        # print(f'save to {os.path.join(save_dir, f"{session}.png")}')
    print(f'exp: {root}, accuracy: {np.mean(accs)}({np.std(accs)}), error: {np.mean(errors)}({np.std(errors)})')
