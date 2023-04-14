from dataset.unsw import UNSWDataset
from classifier import byteiot, tmc, big_data, audi
# -- coding: utf-8 --


def preprocess_private_dataset():
    private_dataset = UNSWDataset()
    private_dataset.run_tshark()
    # private_dataset.get_entropy_feature()


if __name__ == '__main__':
    # preprocess_private_dataset()
    audi_classifier = audi.AuDIClassifier()
    print("完成初始化")
    audi_classifier.get_archived_dataset('UNSW')
    print("完成数据集转换")
    #
    audi_classifier.train_on_unsw_dataset()
    print('traning phase completed')

    # byteiot_classifier = byteiot.ByteIoTClassifier()
    # # byteiot_classifier.get_archived_dataset('Private')
    # byteiot_classifier.get_archived_dataset('UNSW')
    # byteiot_classifier.train_on_private_dataset()
    # print('training phase completed')
    #
    # byteiot_classifier.test(UNSWDataset(), './byteiot-test.pkl')
    # print('test phase completed')

