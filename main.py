# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
# from keras.datasets import mnist
import numpy
import numpy as np
from scipy.io import loadmat
import scipy.stats as st

data_name = "./mnist_data.mat"  # 'trX', 'trY', 'tsX', 'tsY'


def naive_bayes_classifier(mat_data):
    # 处理数据， 计算每一张图片像素的均值和标准差，用于下一步计算
    tr_mean = np.mean(data['trX'], axis=1)
    tr_std = np.std(data['trX'], axis=1)
    ts_mean = np.mean(data['tsX'], axis=1)
    ts_std = np.std(data['tsX'], axis=1)
    ts_Y = data['tsY']

    print("tr_mean.shape, tr_std.shape, ts_mean.shape, ts_std.shape")
    print(tr_mean.shape, tr_std.shape, ts_mean.shape, ts_std.shape)
    # print(tr_mean)
    # print(tr_std)

    # 数据分类
    trX_new = np.vstack((tr_mean, tr_std)).T
    tsX_new = np.vstack((ts_mean, ts_std)).T

    # 分离7、8, 数据前半部分是7，后半部分是8
    trX_7_new = trX_new[0:6265, ...]
    trX_8_new = trX_new[6265:12116, ...]

    # 图示
    plt.scatter(trX_7_new[:, 0], trX_7_new[:, 1], s=1, alpha=0.5)
    plt.scatter(trX_8_new[:, 0], trX_8_new[:, 1], s=1, alpha=0.5)
    plt.show()

    print("trX_7_new.shape, trX_8_new.shape")
    print(trX_7_new.shape, trX_8_new.shape)

    # tr_7_mean = np.mean(trX_7_new, axis=0)
    # tr_7_std = np.std(trX_7_new, axis=0)

    # 计算标签7像素均值的高斯函数
    m_mean_7, m_std_7 = st.norm.fit(trX_7_new[:, 0])
    # print("m_mean_7, m_std_7")
    # print(m_mean_7, m_std_7)
    # 计算标签7像素方差的高斯函数
    s_mean_7, s_std_7 = st.norm.fit(trX_7_new[:, 1])
    # print("s_mean_7, s_std_7")
    # print(s_mean_7, s_std_7)

    # tr_8_mean = np.mean(trX_8_new, axis=0)
    # tr_8_std = np.std(trX_8_new, axis=0)

    # 计算标签7像素均值的高斯函数
    m_mean_8, m_std_8 = st.norm.fit(trX_8_new[:, 0])
    # print("m_mean_8, m_std_8")
    # print(m_mean_8, m_std_8)
    # 计算标签8像素方差的高斯函数
    s_mean_8, s_std_8 = st.norm.fit(trX_8_new[:, 1])
    # print("s_mean_8, s_std_8")
    # print(s_mean_8, s_std_8)

    # 根据贝叶斯计算标签为7或者8的概率
    # p_7 = p(mean|7) x p(std|7) x p(7)
    # p_8 = p(mean|8) x p(std|8) x p(8)

    # test
    right_sum = 0
    wrong_sum = 0
    for sample_i in range(2002):
        # sample_i = 200
        sample = tsX_new[sample_i, :]
        # print(sample, ts_Y[0][sample_i])
        p_7 = st.norm.pdf(sample[0], m_mean_7, m_std_7) * st.norm.pdf(sample[1], s_mean_7, s_std_7) * (6265 / 12116)
        p_8 = st.norm.pdf(sample[0], m_mean_8, m_std_8) * st.norm.pdf(sample[1], s_mean_8, s_std_8) * (5851 / 12116)
        if p_7 > p_8:
            predict = 0
        else:
            predict = 1

        if ts_Y[0][sample_i] == predict:
            right_sum += 1
        else:
            wrong_sum += 1

    print("right_sum, wrong_sum, right_rate")
    print(right_sum, wrong_sum, right_sum / 2002)

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape)
    # plt.imshow(data['trX'][0].reshape((28, 28)))
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载数据
    data = loadmat(data_name)
    # print(data['trX'].shape, data['trY'].shape)
    # print(data['trY'])
    # print(data['tsX'].shape, data['tsY'].shape)
    naive_bayes_classifier(data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
