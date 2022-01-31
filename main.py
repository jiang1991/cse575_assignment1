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


def naive_bayes_classifier(data):
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

    # 图示，数据可视化
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
        # 分别求出7的概率与8的概率作比较
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


# sigmoid函数
def sigmoid(in_x):  # sigmoid函数
    return 1.0 / (1 + np.exp(-in_x))


# 梯度上升求最优参数
def grad_ascent(data, label):
    m, n = np.shape(data)
    alpha = 0.00001  # 设置梯度的阀值，该值越大梯度上升幅度越大
    max_cycles = 300000  # 设置迭代的次数，一般看实际数据进行设定，有些可能200次就够了
    weights = np.ones((n, 1))  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。

    for k in range(max_cycles):
        h = sigmoid(data * weights)
        error = (label - h)  # 求导后差值
        weights = weights + alpha * data.transpose() * error  # 迭代更新权重
    return weights


# 逻辑回归
def logistic_regression(data):
    # 处理数据， 计算每一张图片像素的均值和标准差，用于下一步计算
    tr_mean = np.mean(data['trX'], axis=1)
    tr_std = np.std(data['trX'], axis=1)
    trY = data['trY'].T
    ts_mean = np.mean(data['tsX'], axis=1)
    ts_std = np.std(data['tsX'], axis=1)
    tsY = data['tsY'].T

    print("tr_mean.shape, tr_std.shape, ts_mean.shape, ts_std.shape")
    print(tr_mean.shape, tr_std.shape, ts_mean.shape, ts_std.shape)

    # 数据分类
    trX_new = np.vstack((tr_mean, tr_std)).T
    tsX_new = np.vstack((ts_mean, ts_std)).T

    #
    weight = grad_ascent(np.mat(trX_new), trY)
    print("weight.shape")
    print(weight.shape)

    # 进行预测，并将预测评分存入 predict 列中
    predict = []
    test = np.mat(trX_new)
    for i in test:
        sig = sigmoid(i*np.mat(weight))
        # print("sig.shape")
        # print(sig.shape)
        if sig <= 0.6:
            predict.append('0')
        else:
            predict.append('1')

    # 计算预测准确率
    right_sum = 0
    wrong_sum = 0
    for i in range(12116):
        if int(trY[i][0]) == int(predict[i]):
            right_sum += 1
        else:
            wrong_sum += 1

    print("right_sum, wrong_sum, right_rate")
    print(right_sum, wrong_sum, right_sum / 12116)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载数据
    mat_data = loadmat(data_name)
    # print(data['trX'].shape, data['trY'].shape)
    # print(data['trY'])
    # print(data['tsX'].shape, data['tsY'].shape)
    # naive_bayes_classifier(mat_data)
    logistic_regression(mat_data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
