# -*- coding: utf-8 -*-
import numpy as np
def distance(vecA, vecB):
    '''计算vecA与vecB之间的欧式距离的平方
    input:  vecA(mat)A点坐标
            vecB(mat)B点坐标
    output: dist[0, 0](float)A点与B点距离的平方
    '''
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]
# TODO Try to implement a better randCent function
def randCent(data, k):
    #k-means++主要是这里的区别
    '''随机初始化聚类中心
    input:  data(mat):训练数据
            k(int):类别个数
    output: centroids(mat):聚类中心
    '''
    n = np.shape(data)[1]  # 属性的个数
    #centroids = np.mat(np.zeros((k, n)))  # 初始化k个聚类中心
    #
    #TODO Try to  randCent function
    #
    np.random.seed(100)
    centroids = np.mat(np.random.rand(k,n))
    return centroids

    return centroids
def kmeans(data, k, centroids):
    '''根据KMeans算法求解聚类中心
    input:  data(mat):训练数据
            k(int):类别个数
            centroids(mat):随机初始化的聚类中心
    output: centroids(mat):训练完成的聚类中心
            subCenter(mat):每一个样本所属的类别
    '''
    m, n = np.shape(data)  # m：样本的个数，n：特征的维度
    subCenter = np.mat(np.zeros((m, 2)))  # 初始化每一个样本所属的类别
    change = True  # 判断是否需要重新计算聚类中心
    CurDist = 0.000001
    while change == True:
        change = False  # 重置
        #
        # TODO update labels of each feature
        #
        TmpCentroids = np.mat(np.zeros((k, n)))  # 初始化每一个样本所属的类别
        NumCentroids = np.mat(np.zeros((k, 1)))  # 初始化每一个样本所属的类别

        # 遍历每个样本
        for i in range(m):
            # 判断是否需要改变自身所属的类别
            minIndex = 0
            minDist = np.inf
            # 遍历每个聚类中心
            for j in range(k):
                if distance(data[i,], centroids[j,]) < minDist:
                    minDist = distance(data[i,], centroids[j,])
                    minIndex = j

            if subCenter[i, 1] < minDist:  # 需要改变
                change = True
                subCenter[i,] = np.mat([minIndex, minDist])
        # 重新计算聚类中心
        # TODO complete updatecentersfunctions.
        #
        #     // TODO Write a terminate function.
        #     // helper funtion: check_convergence
        #
            # 累加属于某一聚类中心的数据点，将来重新计算新聚类中心使用
            minIndex = int(subCenter[i, 0])
            TmpCentroids[minIndex] += data[i]
            NumCentroids[minIndex] += 1

        # 重新计算k个聚类中心
        for i in range(k):
            if NumCentroids[i] == 0:
                continue
            centroids[i] = TmpCentroids[i] / NumCentroids[i]

        tag, CurDist = check_convergence(data, CurDist)
        if tag:
            break
    return centroids, subCenter
def save_result(file_name, source):
    '''保存source中的结果到file_name文件中
    input:  file_name(string):文件名
            source(mat):需要保存的数据
    output:
    '''
    m, n = np.shape(source)
    f = open(file_name, "w")
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()

def check_convergence(data, PreDist):
    m, n = np.shape(data)  # m：样本的个数，n：特征的维度
    Dist = 0

    # 计算距离和
    for i in range(m):
        Dist += data[i,1]**2

    # 如果上次和这次的距离和变化小于1%，则认为收敛
    if abs(PreDist - Dist) // float(PreDist) < 0.01:
        return True, Dist
    return False, Dist