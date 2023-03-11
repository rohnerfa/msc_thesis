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