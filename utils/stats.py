import os
import glob

import pandas as pd
import dtw
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from scipy.signal import find_peaks, peak_widths
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
    data = data.copy()
    win_size = 10
    data = data.select_dtypes(include='number')
    data['score_delta'] = data['score'].diff()
    data['score_change_rate'] = data['score_delta'] / data['score'].shift(1)
    smooth = data.rolling(window=win_size).mean()
    smooth['arousal'] = data['arousal']
    # out = 'arousal_delta'

    smooth['arousal_delta'] = data['arousal'].diff()
    # corr = numerics.corr()[out]
    corr = smooth.corr()['arousal_delta'].sort_values(ascending=False)

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


def find_inflection_points(y, t=12, prominence=0, distance=1):
    # Compute the first derivative (gradient)
    dy = np.diff(y)

    # Compute smoothed gradient using a moving average of t steps
    smoothed_dy = np.convolve(dy, np.ones(t) / t, mode='valid')
    gradient_signs = np.sign(smoothed_dy)  # 1 for increase, -1 for decrease, 0 for flat

    # Find peaks (local maxima) and valleys (local minima) as inflection points
    peaks, peak_props = find_peaks(y, prominence=prominence, distance=distance)
    valleys, valley_props = find_peaks(-y, prominence=prominence, distance=distance)
    inflection_points = np.sort(np.concatenate((peaks, valleys)))

    # Compute peak and valley widths
    peak_widths_vals = peak_widths(y, peaks, rel_height=0.5)[0]
    valley_widths_vals = peak_widths(-y, valleys, rel_height=0.5)[0]

    # Determine start and end points of peak/valley areas using their widths
    peak_start = (peaks - peak_widths_vals.astype(int) // 2).clip(min=0)
    peak_end = (peaks + peak_widths_vals.astype(int) // 2).clip(max=len(y) - 1)
    valley_start = (valleys - valley_widths_vals.astype(int) // 2).clip(min=0)
    valley_end = (valleys + valley_widths_vals.astype(int) // 2).clip(max=len(y) - 1)

    # Create a mask for excluded points
    excluded_mask = np.zeros(len(y), dtype=bool)
    for start, end in zip(peak_start, peak_end):
        excluded_mask[start:end + 1] = True
    for start, end in zip(valley_start, valley_end):
        excluded_mask[start:end + 1] = True

    # Find gradient change start points
    change_points = np.where(np.diff(gradient_signs) != 0)[0] + t  # Adjust index to original scale
    valid_change_points = [cp for cp in change_points if not excluded_mask[cp]]

    # Concatenate inflection points and valid change points
    inflection_points = np.sort(np.concatenate((inflection_points, valid_change_points))).astype(int)

    return inflection_points


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


def compute_inflection_f1(predict_inflections, arousal_inflections):
    unused_predicts = set(predict_inflections)
    correct_peak_cnt = 0

    for arousal_inflection in arousal_inflections:
        # Find candidate predictions within the window
        candidates = [p for p in unused_predicts if abs(p - arousal_inflection) < 12]
        if candidates:
            # Pick the closest one
            best_match = min(candidates, key=lambda p: abs(p - arousal_inflection))
            correct_peak_cnt += 1
            unused_predicts.remove(best_match)

    precision = correct_peak_cnt / len(predict_inflections) if len(predict_inflections) > 0 else 0
    recall = correct_peak_cnt / len(arousal_inflections) if len(arousal_inflections) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def find_combat_points(data):
    data['score_delta'] = data['score'].diff()
    # z = zscore(data['score_delta']).fillna(0)
    # print(data['game'].unique()[0], data[['score', 'score_delta']])
    data['event_flag'] = (data['score_delta'] != 0).astype(int) # (z > 1).astype(int)
    # Find the combat intervals
    event_indices = data.index[data['event_flag'] == 1].to_numpy() - data.index[0]

    # Group contiguous indices (max gap of 2 = 0.5 sec, allows for a gap < 3 sec at 4 FPS)
    combat_groups = []
    current_group = [event_indices[0]]

    for i in range(1, len(event_indices)):
        if event_indices[i] - event_indices[i - 1] <= 8:  # 3 seconds at 4 FPS = 12 frames
            current_group.append(event_indices[i])
        else:
            combat_groups.append(current_group)
            current_group = [event_indices[i]]
    combat_groups.append(current_group)

    # Classify each combat segment
    combat_events = []
    for group in combat_groups:
        duration = len(group) / 4  # seconds
        if duration < 3:
            mid_idx = group[len(group) // 2]
            combat_events.append(mid_idx)
        else:
            combat_events.append(group[0])
            combat_events.append(group[-1])

    # print(f"{data['game']}: {combat_events}")
    return combat_events


def compute_roi_f1(predict_rois, arousal_rois, total_length):
    pred_mask = np.zeros(total_length, dtype=bool)
    true_mask = np.zeros(total_length, dtype=bool)

    for s, e in predict_rois:
        pred_mask[s:e] = True
    for s, e in arousal_rois:
        true_mask[s:e] = True

    tp = np.sum(np.logical_and(pred_mask, true_mask))
    fp = np.sum(np.logical_and(pred_mask, ~true_mask))
    fn = np.sum(np.logical_and(~pred_mask, true_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def inflection_comparison_manual(data, game, roi=False):
    sessions = data['session_id'].unique()
    avg_inf_cnt = {
        'TinyCars': 30,
        'Solid': 38,
        'ApexSpeed': 31,
        'Heist!': 35,
        'Shootout': 35,
        'TopDown': 34,
        "Run'N'Gun": 36,
        "Pirates!": 28,
        "Endless": 25
    }
    ip_cnt = avg_inf_cnt[game]

    uniform_f1s = []
    random_f1s = []
    manual_f1s = []
    te_unit = []
    te_unia = []
    te_randt = []
    te_randa = []
    te_rulet = []
    te_rulea = []
    for session in sessions:
        session_data = data[data['session_id'] == session].copy()
        arousal = session_data['arousal']
        inflection_points = find_inflection_points(arousal)
        # ip_cnt = len(inflection_points)

        uniform_idx = np.linspace(0, len(arousal)-1, ip_cnt, dtype=int)
        random_idx = np.sort(np.random.choice(np.arange(len(arousal)), ip_cnt, replace=False))
        rule_based_idx = np.sort(find_combat_points(session_data))

        if roi:
            uniform_rois = build_roi_from_inflections(uniform_idx, end=len(arousal))
            random_rois = build_roi_from_inflections(random_idx, end=len(arousal))
            rule_based_rois = build_roi_from_inflections(rule_based_idx, end=len(arousal))
            arousal_rois = build_roi_from_inflections(inflection_points, end=len(arousal))
            te_unit.append(compute_time_efficiency(len(arousal), uniform_rois))
            te_unia.append(compute_time_efficiency(arousal_rois, uniform_rois))
            te_randt.append(compute_time_efficiency(len(arousal), random_rois))
            te_randa.append(compute_time_efficiency(arousal_rois, random_rois))
            te_rulet.append(compute_time_efficiency(len(arousal), rule_based_rois))
            te_rulea.append(compute_time_efficiency(arousal_rois, rule_based_rois))
            # print(f"{session_data['game'].unique()[0]} true rois:{arousal_rois}, rule_rois: {rule_based_rois}")

            f1_uniform = compute_roi_f1(uniform_rois, arousal_rois, len(arousal))
            f1_random = compute_roi_f1(random_rois, arousal_rois, len(arousal))
            f1_manual = compute_roi_f1(rule_based_rois, arousal_rois, len(arousal))
        else:
            f1_uniform = compute_inflection_f1(uniform_idx, inflection_points)
            f1_random = compute_inflection_f1(random_idx, inflection_points)
            f1_manual = compute_inflection_f1(rule_based_idx, inflection_points)
        uniform_f1s.append(f1_uniform)
        random_f1s.append(f1_random)
        manual_f1s.append(f1_manual)
    uniform_f1mean = np.mean(uniform_f1s)
    uniform_f1std = np.std(uniform_f1s)
    random_f1mean = np.mean(random_f1s)
    random_f1std = np.std(random_f1s)
    manual_f1mean = np.mean(manual_f1s)
    manual_f1std = np.std(manual_f1s)
    print(f'game: {game}')
    print(f'uniform_f2s: {uniform_f1mean}/{uniform_f1std}')
    print(f'random_f2s: {random_f1mean}/{random_f1std}')
    print(f'manual_f2s: {manual_f1mean}/{manual_f1std}')
    print('-'*50)

    result = {
        'Uniform': {
            'f1_mean': uniform_f1mean,
            'f1_std': uniform_f1std,
            'per_total': np.mean(te_unit),
            'per_arousal': np.mean(te_unia),
        },
        'Random': {
            'f1_mean': random_f1mean,
            'f1_std': random_f1std,
            'per_total': np.mean(te_randt),
            'per_arousal': np.mean(te_randa),
        },
        'Rule_based': {
            'f1_mean': manual_f1mean,
            'f1_std': manual_f1std,
            'per_total': np.mean(te_rulet),
            'per_arousal': np.mean(te_rulea),
        }
    }

    return result


def build_roi_from_inflections(inflections, window=10, end=400):
    rois = []
    for idx in inflections:
        lower_bound = idx - window if idx - window > 0 else 0
        upper_bound = idx + window if idx + window < end else end
        if len(rois) == 0 or lower_bound > rois[-1][1]:
            if lower_bound < 0:
                rois.append([0, 2 * window])
            else:
                rois.append([lower_bound, upper_bound])
        else:
            rois[-1][1] = max(upper_bound, rois[-1][1])
    return rois


def inflection_comparison(root, show=False, epoch=False, roi=False):
    if epoch:
        log_dirs = glob.glob(os.path.join(root, 'test_epc[5-9][0-9]*'))  # recent 10 epochs (epoch 60)
    else:
        log_dirs = glob.glob(os.path.join(root, 'test_*'))
    # print(f'root: {root} log_dirs: {log_dirs}')
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

    # print(log_dict)
    f1_scores = []
    te_pt = []
    te_pa = []
    te_at = []
    for session, tags in log_dict.items():
        arousal_path = tags['arousal']
        predict_path = tags['predict']

        arousal_event_files = glob.glob(os.path.join(arousal_path, "events.out.tfevents.*"))
        predict_event_files = glob.glob(os.path.join(predict_path, "events.out.tfevents.*"))
        _, arousal_summary = read_scalar_summary(arousal_event_files[0])
        _, predict_summary = read_scalar_summary(predict_event_files[0])

        predict_peaks, predict_valleys = find_significant_peaks_and_valleys(predict_summary, threshold=0, prominence=0.01)
        predict_inflections = np.concatenate([predict_peaks, predict_valleys])
        arousal_inflections = find_inflection_points(arousal_summary)

        if roi:
            predict_rois = build_roi_from_inflections(np.sort(predict_inflections), end=len(predict_summary))
            arousal_rois = build_roi_from_inflections(np.sort(arousal_inflections), end=len(arousal_summary))
            te_pt.append(compute_time_efficiency(len(arousal_summary), predict_rois))
            te_pa.append(compute_time_efficiency(arousal_rois, predict_rois))
            te_at.append(compute_time_efficiency(len(arousal_summary), arousal_rois))

            for s, e in predict_rois:
                plt.axvspan(s, e, color='green', alpha=0.2)
            for s, e in arousal_rois:
                plt.axvspan(s, e, color='red', alpha=0.2)

            f1_score = compute_roi_f1(predict_rois, arousal_rois, len(arousal_summary))
        else:
            f1_score = compute_inflection_f1(predict_inflections, arousal_inflections)
        f1_scores.append(f1_score)

        plt.plot(arousal_summary, label='arousal (ground truth)')
        plt.plot(predict_summary, color='gray', linestyle='--', alpha=0.5, label='predicted arousal')
        plt.plot(arousal_inflections, arousal_summary[arousal_inflections], "r*", label='true peaks')
        plt.plot(predict_inflections, predict_summary[predict_inflections], "g*", alpha=0.5, label='predicted peaks')

        for inflection in predict_inflections:
            plt.vlines(inflection, arousal_summary[inflection], predict_summary[inflection], colors='green', linestyles=':', alpha=0.5)

        plt.title(f"{root.split('/')[-1]}_{session}")
        plt.legend()

        save_dir = os.path.join('/', *root.split('/')[:-1], 'peak', root.split('/')[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, f'{session}.png'))
        if show:
            plt.show()
    print(f'exp: {root}, f1 score: {np.mean(f1_scores)}({np.std(f1_scores)})')
    result = {
        'f1_score': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'per_total': np.mean(te_pt),
        'per_arousal': np.mean(te_pa),
        'gt_per_total': np.mean(te_at),
    }
    return result


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

def compute_time_efficiency_by_log(root, player, session):
    log_file = os.path.join(root, player, f'{player}_topdown_{session}.csv')
    roi_file = os.path.join(root, player, f'{player}_{session}', 'roi.csv')

    log = pd.read_csv(log_file)
    roi = pd.read_csv(roi_file)

    total_duration = log['timeStamp']
    clip_total_duation = 0
    for _, row in roi.iterrows():
        start = row['start']
        end = row['end']
        clip_total_duation += end - start

    time_efficiency = clip_total_duation / total_duration
    return time_efficiency, total_duration, clip_total_duation


def compute_time_efficiency(total, rois):
    clip_total_duation = 0
    total_duration = 0
    if type(total) != int:
        for roi in total:
            start = roi[0]
            end = roi[1]

            total_duration += end - start + 1
    else:
        total_duration = total
    for roi in rois:
        start = roi[0]
        end = roi[1]

        clip_total_duation += end - start + 1

    time_efficiency = clip_total_duation / total_duration
    return time_efficiency