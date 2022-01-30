# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
# from keras.datasets import mnist
import numpy as np
from scipy.io import loadmat

data_name = "./mnist_data.mat"  # 'trX', 'trY', 'tsX', 'tsY'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = loadmat(data_name)
    print(data['trX'].shape, data['trY'].shape)
    print(data['tsX'].shape, data['tsY'].shape)

    tr_mean = np.mean(data['trX'], axis=1)
    tr_cov = np.cov(data['trX'][0])
    ts_mean = np.mean(data['tsX'], axis=1)
    ts_cov = np.cov(data['tsX'][0])

    print(tr_mean.shape, tr_cov.shape, ts_mean.shape, ts_cov.shape)
    print(tr_cov)
    print(ts_cov)

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape)
    # plt.imshow(data['trX'][0].reshape((28, 28)))
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
