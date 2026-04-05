from copy import deepcopy
import pickle
import logging

class BestCkptMeasure:
    def __init__(self):
        self._training_loss = 987654321.0
        self._mpe = 987654321.0  # mean pixel error
        self._map = -1.0
        self._epoch = -1

    def set_map(self, map):
        self._map = map

    def get_map(self):
        return deepcopy(self._map)

    def set_training_loss(self, training_loss):
        self._training_loss = training_loss

    def get_training_loss(self):
        return deepcopy(self._training_loss)

    def set_mpe(self, mpe):
        self._mpe = mpe

    def get_mpe(self):
        return deepcopy(self._mpe)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def get_epoch(self):
        return deepcopy(self._epoch)

    def write_pkl(self, file_name):
        try:
            pickle.dump(self.__dict__, open(file_name, 'wb'))
        except Exception as e:
            logging.info(e)

    def read_pkl(self, file_name):
        try:
            read_dict = pickle.load(open(file_name, 'rb'))
            self.__dict__.update(read_dict)
        except Exception as e:
            logging.info(e)
