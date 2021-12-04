import numpy as np
from random import random
from KMeans import kmeans, distance, save_result
FLOAT_MAX = 1e100  # 设置一个较大的值作为初始化的最小的距离
def nearest(point, cluster_centers):
    '''计算point和cluster_centers之间的最小距离
    input:  point(mat):当前的样本点
            cluster_centers(mat):当前已经初始化的聚类中心
    output: min_dist(float):点point和当前的聚类中心之间的最短距离
    '''
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i,])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist
def get_centroids(data, k):
    '''KMeans++的初始化聚类中心的方法
    input:  points(mat):样本
            k(int):聚类中心的个数
    output: cluster_centers(mat):初始化后的聚类中心
    '''
    #
    # TODO Try to implement a better initialization function
    #
    #
    np.random.seed(100)
    m, n = np.shape(data)  # m：样本的个数，n：特征的维度
    first = np.random.choice(a=m, size=1, replace=False, p=None)
    centroids = np.mat(np.zeros((k, n)))  # 初始化k个聚类中心
    centroids[0] = data[first]
    #
    # TODO Try to  randCent function
    #
    added = []
    added.append(first)
    # 因为是根据前面所有的聚类中心才能确定下一个聚类中心，所以一共遍历k-1次
    for i in range(1,k):
        index = None
        MaxDist = -np.inf
        # 遍历每个数据点
        for j in range(m):
            if j in added:
                continue
            Dist = 0
            # 遍历每个聚类中心
            for k in range(i):
                Dist += distance(centroids[k], data[j])
            if Dist > MaxDist:
                index = j
                MaxDist = Dist
                added.append(index)
        centroids[i] = data[index]

    return centroids
