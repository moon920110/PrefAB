import os
import glob

import pandas as pd
import dtw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator, CubicSpline

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


def inflection_comparison(root, show=False, epoch=False):
    if epoch:
        log_dirs = glob.glob(os.path.join(root, 'test_epc[5-9][0-9]*'))  # recent 10 epochs (epoch 60)
    else:
        log_dirs = glob.glob(os.path.join(root, 'test_*'))
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

    print(log_dict)
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
        arousal_inflections = np.concatenate([arousal_peaks, arousal_valleys])
        predict_inflections = np.concatenate([predict_peaks, predict_valleys])

        correct_peak_cnt = 0
        error_peak_cnt = 0
        for arousal_inflection in arousal_inflections:
            # find nearest peak in predict within +-10 distance
            nearest_peak = np.argmin(np.abs(arousal_inflection - predict_inflections))
            if np.abs(arousal_inflection - predict_inflections[nearest_peak]) < 10:  # half of distance
                correct_peak_cnt += 1
        accs.append(correct_peak_cnt / len(predict_inflections))

        for predict_inflection in predict_inflections:
            # find nearest peak in predict within +-10 distance
            nearest_peak = np.argmin(np.abs(predict_inflection - arousal_inflections))
            if np.abs(predict_inflection - arousal_inflections[nearest_peak]) >= 10:
                error_peak_cnt += 1
        errors.append(error_peak_cnt / len(predict_inflections))

        plt.plot(arousal_summary, label='arousal (ground truth)')
        plt.plot(predict_summary, color='gray', linestyle='--', alpha=0.5, label='predicted arousal')
        plt.plot(arousal_inflections, arousal_summary[arousal_inflections], "r*", label='true peaks')
        # plt.plot(arousal_valleys, arousal_summary[arousal_valleys], "r*")
        plt.plot(predict_inflections, predict_summary[predict_inflections], "g*", alpha=0.5, label='predicted peaks')
        # plt.plot(predict_valleys, predict_summary[predict_valleys], "g*", alpha=0.5)

        for inflection in predict_inflections:
            plt.vlines(inflection, arousal_summary[inflection], predict_summary[inflection], colors='green', linestyles=':', alpha=0.5)
            # plt.vlines(valley, arousal_summary[valley], predict_summary[valley], colors='green', linestyles=':', alpha=0.5)

        plt.title(session)
        plt.legend()

        save_dir = os.path.join(root.split('/')[0], 'peak', root.split('/')[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # print(f'make dir: {save_dir}')

        plt.savefig(os.path.join(save_dir, f'{session}.png'))
        if show:
            plt.show()
        # print(f'save to {os.path.join(save_dir, f"{session}.png")}')
    print(f'exp: {root}, accuracy: {np.mean(accs)}({np.std(accs)}), error: {np.mean(errors)}({np.std(errors)})')


# TODO: uniform sample/random sample로 interpolation한 것과 결과 비교해볼것, 평균적으로 inflection이 얼마나 발생하는 지 계산해볼것
def reconstruct_state_via_interpolation(root, show=False, epoch=False):
    if epoch:
        log_dirs = glob.glob(os.path.join(root, 'test_epc[5-9][0-9]*'))  # recent 10 epochs (epoch 60)
    else:
        log_dirs = glob.glob(os.path.join(root, 'test*'))
    log_dict = {}
    window_size = 6
    sample_size = 10

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

    for session, tags in log_dict.items():
        arousal_path = tags['arousal']
        predict_path = tags['predict']

        arousal_event_files = glob.glob(os.path.join(arousal_path, "events.out.tfevents.*"))
        predict_event_files = glob.glob(os.path.join(predict_path, "events.out.tfevents.*"))
        _, arousal_summary = read_scalar_summary(arousal_event_files[0])
        _, predict_summary = read_scalar_summary(predict_event_files[0])

        predict_peaks, predict_valleys = find_significant_peaks_and_valleys(predict_summary, threshold=0,
                                                                            prominence=0.01, distance=20)
        predict_inflections = sorted(np.concatenate([predict_peaks, predict_valleys]))

        gt_arousals = [arousal_summary[0]]
        gt_indices = [[0]]
        for predict_peak in predict_inflections:
            front_offset = window_size if predict_peak - window_size > 0 else 0
            end_offset = window_size if predict_peak + window_size < len(arousal_summary) else 0
            gt_indices.append(np.arange(predict_peak - front_offset, predict_peak + end_offset))
            gt_arousals.append(arousal_summary[predict_peak - front_offset:predict_peak + end_offset])
        gt_arousals.append(arousal_summary[-1])
        gt_indices.append([len(arousal_summary) - 1])

        gt_indices = np.hstack(gt_indices)
        gt_arousals = np.hstack(gt_arousals)

        unique_indices, unique_values = np.unique(gt_indices, return_index=True)
        gt_indices = gt_indices[unique_values]
        gt_arousals = gt_arousals[unique_values]

        pchip_interpolator = PchipInterpolator(gt_indices, gt_arousals)

        full_range_indices = np.linspace(0, len(arousal_summary), len(arousal_summary))

        pchip_values = pchip_interpolator(full_range_indices)

        plt.plot(arousal_summary, label='arousal (ground truth)')
        plt.plot(gt_indices, gt_arousals, 'o', label='Sampled points', alpha=0.3)
        plt.plot(predict_inflections, arousal_summary[predict_inflections], 'x', label='Predicted peaks')
        plt.plot(full_range_indices, pchip_values, label='Pchip interpolation', alpha=0.5)
        plt.legend()
        plt.title(session)

        save_dir = os.path.join(root.split('/')[0], 'reconstruct', root.split('/')[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # print(f'make dir: {save_dir}')

        plt.savefig(os.path.join(save_dir, f'{session}.png'))
        if show:
            plt.show()
