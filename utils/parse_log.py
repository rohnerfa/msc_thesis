from pathlib import Path

def parse_tensorboard_log(path_to_log):
    
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd
    import numpy as np

    event_acc = EventAccumulator(path_to_log)
    event_acc.Reload()
    #tags = event_acc.Tags()['scalars']
    
    df2 = pd.DataFrame()
    w_times, step_nums, val_loss = zip(*event_acc.Scalars('val_loss'))
    w_times = np.array(w_times) - np.array(w_times)[0]
    df2['steps'], df2['val_loss'], df2['wall_time'] = step_nums, val_loss, w_times
    return df2

def get_latest_path(path):
    visible_files = [
        int(file.name.split('_')[1]) for file in Path(path).iterdir() if not file.name.startswith(".")
    ]
    visible_files.sort()
    return path + '/version_' + str(visible_files[-1])

def get_log_path(path):
    latest_path = get_latest_path(path)
    arr = [str(file) for file in Path(latest_path).iterdir() if 'tfevents' in file.name]
    return arr[0]

def parse_latest(path):
    log_path = get_log_path(path)
    return parse_tensorboard_log(log_path)