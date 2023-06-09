from dataset import unsw, yourthings, private

import pickle

import numpy as np


class Classifier(object):
    def __init__(self):
        self._dataset = {
            'UNSW': unsw.UNSWDataset(),
            'Yourthings': yourthings.YourthingsDataset(),
            'Private': private.PrivateDataset()
        }
        self.selected_features = []
        self.tag = 'base'
        self.model = None
        self.preprocessor = None

    def get_dataset(self, raw_dataset, generator):
        raise NotImplementedError

    def train_model(self, dataset, training_set_archive):
        raise NotImplementedError

    @staticmethod
    def get_training_dataset(dataset, training_set_archive):
        with open(training_set_archive, 'rb') as f:
            train_set = pickle.load(f)
        x, y = [], []
        for address, features in train_set.items():
            for feature in features:
                x.append(feature)
                y.append(dataset.label_map[address])
        x_train, y_train = np.array(x), np.array(y)
        # print(x_train.shape, y_train.shape)
        return x_train, y_train

    def get_archived_dataset(self, dataset_tag, train_range=None, test_range=None):
        if dataset_tag not in self._dataset:
            raise ValueError("Unsupported Dataset")
        raw_dataset = self._dataset[dataset_tag]
        print(raw_dataset)
        if not train_range:
            train_range = raw_dataset.default_training_range['train']
        generator = raw_dataset.data_generator(**train_range, features=self.selected_features)
        # 进行生成生成器，generator没有执行上一行的函数
        print(generator)
        # generator被使用，data_generator开始执行
        training_set = self.get_dataset(raw_dataset, generator)
        with open(self.tag + '-train.pkl', 'wb') as f:
            pickle.dump(training_set, f)

        if not test_range:
            test_range = raw_dataset.default_training_range['test']
        generator = raw_dataset.data_generator(**test_range, features=self.selected_features)
        test_set = self.get_dataset(raw_dataset, generator)
        with open(self.tag + '-test.pkl', 'wb') as f:
            pickle.dump(test_set, f)

    def train_on_unsw_dataset(self):
        training_set = self.tag + '-train.pkl'
        self.train_model(self._dataset['UNSW'], training_set)

    def train_on_yourthings_dataset(self):
        training_set = self.tag + '-train.pkl'
        self.train_model(self._dataset['Yourthings'], training_set)

    def train_on_private_dataset(self):
        training_set = self.tag + '-train.pkl'
        self.train_model(self._dataset['Private'], training_set)

    def test(self, dataset, test_set_archive=None):
        true_count, false_count = 0, 0
        with open(test_set_archive, 'rb') as f:
            test_set = pickle.load(f)
        for address, features in test_set.items():
            for feature in features:
                x_test = np.array([feature])
                if self.preprocessor:
                    x_test = self.preprocessor(x_test)
                y_predict = self.model.predict(x_test)
                if y_predict[0] == dataset.label_map[address]:
                    true_count += 1
                else:
                    false_count += 1
            print(true_count, true_count + false_count, true_count / (true_count + false_count))
        accuracy = true_count / (true_count + false_count)
        print(accuracy)
        return accuracy


