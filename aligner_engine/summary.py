import pickle
import logging


class TrainSummary:
    def __init__(self):
        self.tr_by_epoch = {}
        self.tr_by_iter = {}
        self.va_by_epoch = {}
        self.va_by_iter = {}
        self.update_model_epoch = []
        self.class_index = {}
        self.class_name = {}

    def reset(self):
        self.tr_by_epoch = {}
        self.tr_by_iter = {}
        self.va_by_epoch = {}
        self.va_by_iter = {}
        self.update_model_epoch = []

    def set_class(self, class_idx: dict, class_name: dict):
        self.class_name = class_name  # DatasetSummary class_name
        self.class_index = class_idx  # DatasetSummary class_index

    def add_tr_iter_result(self, iter, info: dict):
        self.tr_by_iter[iter] = info

    def add_tr_epoch_result(self, epoch, info: dict):
        self.tr_by_epoch[epoch] = info

    def add_va_iter_result(self, iter, info: dict):
        self.va_by_iter[iter] = info

    def add_va_epoch_result(self, epoch, info: dict):
        self.va_by_epoch[epoch] = info

    def add_model_update_epoch(self, epoch):
        self.update_model_epoch.append(epoch)

    def __str__(self):
        epoch_info = list(self.tr_by_epoch.keys())
        return f"EPOCH= {epoch_info}\n" \
               f"\ttr_by_epoch={self.tr_by_epoch}\n" \
               f"\tva_by_epoch={self.va_by_epoch}\n" \
               f"\tupdate_model_epoch={self.update_model_epoch}"

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


class ResultSummary:
    def __init__(self):
        self.data_result: dict = {}  # inference_result
        self.data_label: dict = {}  # ground truth
        self.class_index: dict = {}
        self.class_name: dict = {}
        self.loss: dict = {}
        self.aps: dict = {}
        self.map: dict = {}
        self.mpe: dict = {}
        self.mpe_by_class: dict = {}

    # def __len__(self):
    #     return len(self.data_result)

    def reset(self):
        self.data_result = {}
        self.class_index = {}
        self.class_name = {}
        self.loss = {}
        self.aps = {}
        self.map = {}
        self.mpe = {}
        self.mpe_by_class = {}

    def set_class(self, class_idx: dict, class_name: dict):
        self.class_name = class_name  # DatasetSummary class_name
        self.class_index = class_idx  # DatasetSummary class_index

    def add_data_result(self, path, result):
        self.data_result[path] = result

    def add_data_label(self, path, label):
        self.data_label[path] = label

    def get_num_classes(self):
        return len(self.class_index)

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

    def to_json(self):
        raise NotImplementedError

    def summarize_result(self, epoch, loss=None, ap=None, map=None, mpe=None, mpe_by_class=None):
        if loss is not None:
            self.loss[epoch] = loss
        if ap is not None:
            self.aps[epoch] = {}
            for key, item in ap.items():
                self.aps[epoch][key] = item
        if map is not None:
            self.map[epoch] = map
        if mpe is not None:
            self.mpe[epoch] = mpe
        if mpe_by_class is not None:
            self.mpe_by_class[epoch] = mpe_by_class

    def get_metric_by_class(self, epoch, name, class_idx):
        if name != 'mPE':
            raise NotImplementedError
        if epoch not in self.mpe_by_class:
            return None
        return self.mpe_by_class[epoch].get(class_idx)

    def get_metric(self, epoch, name):
        if name == 'loss':
            if epoch in self.loss:
                return self.loss[epoch]
            else:
                return 0
        elif name == 'mAP':
            if epoch in self.map:
                return self.map[epoch]
            else:
                return 0
        elif name == 'mPE':
            if epoch in self.mpe:
                return self.mpe[epoch]
            else:
                return
        else:
            raise NotImplementedError
