import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

    return np.array(steps), np.array(values)