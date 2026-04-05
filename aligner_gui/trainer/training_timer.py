import time


def timestamp2time(timestamp):
    H = int(timestamp / 3600)
    M = int((timestamp % 3600) / 60)
    S = int((timestamp % 60))
    return '%d:%02d:%02d' % (H, M, S)


class TrainingTimer:
    def __init__(self):
        self.total_epoch = 0
        self.start_epoch = 0
        self.epoch_timestamp = []
        self.start_timestamp = 0.0
        self.last_train_iter_len = 0
        self.last_val_iter_len = 0

    def init_timer(self):
        self.total_epoch = 0
        self.start_epoch = 0
        self.epoch_timestamp = []
        self.start_timestamp = 0.0
        self.last_train_iter_len = 0
        self.last_val_iter_len = 0

    def train_start(self, start_epoch, total_epoch):
        self.init_timer()
        cur_time = time.time()

        self.epoch_timestamp.append(cur_time)
        self.start_timestamp = cur_time
        self.total_epoch = total_epoch
        self.start_epoch = start_epoch

    def one_epoch_done(self, current_epoch):
        """
        return:
        - one_epoch_last_proc_time: processing time for last one epoch
        - one_epoch_avg_proc_time: average processing time for one epoch
        - total_proc_time: total processing time until the current epoch since the training starts
        - remaining_proc_time: estimated remaining processing time to the training ends
        """
        cur_time = time.time()
        self.epoch_timestamp.append(cur_time)

        one_epoch_last_proc_time = self.epoch_timestamp[-1] - self.epoch_timestamp[-2]
        total_proc_time = self.epoch_timestamp[-1] - self.epoch_timestamp[0]

        if len(self.epoch_timestamp) >= 5:
            one_epoch_avg_proc_time = (self.epoch_timestamp[-1] - self.epoch_timestamp[-5]) / 4
        else:
            one_epoch_avg_proc_time = total_proc_time / (len(self.epoch_timestamp) - 1)
        remaining_proc_time = one_epoch_avg_proc_time * (self.total_epoch - current_epoch)

        return one_epoch_last_proc_time, one_epoch_avg_proc_time, total_proc_time, remaining_proc_time

    def one_iter_progress(self, phase_type, iter_idx, iter_len, current_epoch):
        if iter_len <= 0:
            return 0.0, 0.0, 0.0

        if phase_type == "train":
            self.last_train_iter_len = iter_len
            current_epoch_units_done = iter_idx + 1
            remaining_current_epoch_units = max(iter_len - current_epoch_units_done, 0) + self.last_val_iter_len
        else:
            self.last_val_iter_len = iter_len
            current_epoch_units_done = self.last_train_iter_len + iter_idx + 1
            remaining_current_epoch_units = max(iter_len - (iter_idx + 1), 0)

        epoch_units = max(self.last_train_iter_len + self.last_val_iter_len, 1)
        completed_epochs = max(current_epoch - self.start_epoch - 1, 0)
        processed_units = completed_epochs * epoch_units + current_epoch_units_done

        processed_time = max(time.time() - self.start_timestamp, 0.0)
        avg_unit_time = processed_time / max(processed_units, 1)

        remaining_epochs = max(self.total_epoch - current_epoch, 0)
        remaining_units = remaining_current_epoch_units + remaining_epochs * epoch_units
        remaining_time = avg_unit_time * remaining_units

        return avg_unit_time, processed_time, remaining_time
