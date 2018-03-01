from scipy import misc
from utils.preprocess import *
import numpy as np
import os


'将生成batch等多个方法封装成一个类'
class Dataprovider():
    '初始化'
    def __init__(self,
                 time,
                 x_tr_path='dataset/train/images/',
                 y_tr_path='dataset/train/labels/',
                 x_test_path = 'dataset/test/images/',
                 y_test_path = 'dataset/test/labels/',
                 ):
        creat_train(time)
        creat_test()
        self.x_tr, self.y_tr = self.read_train(x_tr_path,y_tr_path)
        self.x_test, self.y_test = self.read_test(x_test_path,y_test_path)
        self.batch_offset_tr = 0
        self.batch_offset_test = 0


    def read_train(self, x_tr_path, y_tr_path):
        x_tr = []
        y_tr = []
        list = os.listdir(x_tr_path)
        for index in range(len(list)):
            x_dir = x_tr_path+list[index]
            y_dir = y_tr_path+list[index]
            x_tr.append(self.transfrom(x_dir))
            y_tr.append(np.expand_dims(self.transfrom(y_dir),-1))
        return np.array(x_tr), np.array(y_tr)

    def read_test(self, x_test_path,y_test_path):
        x_test = []
        y_test = []
        list = os.listdir(x_test_path)
        for index in range(len(list)):
            x_dir = x_test_path+list[index]
            y_dir = y_test_path+list[index]
            x_test.append(self.transfrom(x_dir))
            y_test.append(np.expand_dims(self.transfrom(y_dir),-1))
        return np.array(x_test), np.array(y_test)

    def transfrom(self, path):
        return misc.imread(path)

    def get_train(self):
        return self.x_tr, self.y_tr

    def get_test(self):
        return self.x_test, self.y_test


    def next_batch_tr(self, batch_size):
        start = self.batch_offset_tr
        self.batch_offset_tr += batch_size
        if self.batch_offset_tr > self.x_tr.shape[0]:
            # Shuffle the data
            perm = np.arange(self.x_tr.shape[0])
            np.random.shuffle(perm)
            self.x_tr = self.x_tr[perm]
            self.y_tr = self.y_tr[perm]
            start = 0
            self.batch_offset_tr = batch_size
        end = self.batch_offset_tr
        x_tr = self.x_tr[start:end]
        y_tr = self.y_tr[start:end]
        return x_tr, y_tr

    def next_batch_test(self, batch_size):
        start = self.batch_offset_test
        self.batch_offset_test += batch_size
        if self.batch_offset_test > self.x_test.shape[0]:
            # Shuffle the data
            perm = np.arange(self.x_test.shape[0])
            np.random.shuffle(perm)
            self.x_test = self.x_test[perm]
            self.y_test = self.y_test[perm]
            start = 0
            self.batch_offset_test = batch_size
        end = self.batch_offset_test
        x_test = self.x_test[start:end]
        y_test = self.y_test[start:end]
        return x_test, y_test