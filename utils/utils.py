import os
import numpy as np
import h5py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, f1_score
from scipy.stats import kendalltau


def create_new_filename(root, base_name):
    counter = 1
    new_name = base_name

    while os.path.exists(os.path.join(root, new_name)):
        new_name = f'{base_name}_{counter}'
        counter += 1

    return new_name


def read_scalar_summary(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    scalar_event = event_acc.Scalars(event_acc.Tags()['scalars'][0])

    steps = [event.step for event in scalar_event]
    values = [event.value for event in scalar_event]

    unique_steps, indices = np.unique(steps, return_index=True)
    steps = unique_steps
    values = np.array(values)[indices]

    return np.array(steps), np.array(values)


def metric(y_pred, y_true, cutpoints=None, infer_type='ranknet'):
    if infer_type == "ranknet":
        _y_pred = y_pred.detach().cpu().numpy()
        _y_pred = np.where(_y_pred < cutpoints[0], 0, np.where(_y_pred < cutpoints[1], 1, 2))

        _y_true = y_true.detach().cpu().numpy()
        acc = f1_score(_y_true, _y_pred, average='macro')
        cm = confusion_matrix(_y_true, _y_pred, labels=[0, 1, 2])

        return acc, cm

    elif infer_type == "regression":
        return r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    elif infer_type == "kendal_tau":
        _y_pred = y_pred.detach().cpu().numpy()
        _y_true = y_true.detach().cpu().numpy()
        tau, p_val = kendalltau(_y_true, _y_pred, nan_policy='omit')
        return tau, p_val

    else:  # classification (bio-4 classes)
        _y_pred = y_pred.detach().cpu().numpy()
        _y_pred = _y_pred.argmax(axis=-1)
        _y_true = y_true.detach().cpu().numpy()

        acc = accuracy_score(_y_true, _y_pred)
        cm = confusion_matrix(_y_true, _y_pred, labels=[0, 1, 2, 3])
        return acc, cm


def normalize(val: np.ndarray) -> np.ndarray:
    return (val - val.min()) / (val.max() - val.min())


def h5reader(file_path, key):
    with h5py.File(file_path, 'r') as f:
        for k in f[key].keys():
            print(np.array(f[f'{key}/{k}']))


def convert_frame_to_time(frame, fps=4):
    return frame / fps


def convert_time_to_frame(time, fps=4):
    return time * fps