#coding:UTF-8
import PIL.Image as image
import KMeans
import numpy as np
import KMeanspp
def load_data(file_path):
    '''导入数据
    input:  file_path(string):文件的存储位置
    output: data(mat):数据
    '''
    f = open(file_path,"rb")#以二进制方式打开图像文件
    data = []
    im=image.open(f)#导入图片的大小
    m,n=im.size#得到图片的大小
    for i in range(m):
        for j in range(n):
            tmp = []
            x, y, z = im.getpixel((i, j))
            tmp.append(x / 256.0)
            tmp.append(y / 256.0)
            tmp.append(z / 256.0)
            data.append(tmp)
    f.close()
    return np.mat(data),m,n#矩阵形式
if __name__ == "__main__":
    data,m,n = load_data("test_data//trump.jpg")
    k=15
    #print(data)
# 方法1 k-means随机选取初始化随机点
    # 新建一个图片
    f_center  = KMeans.randCent(data, k)
# 方法二
    """
    在数据集中随机选择一个样本点作为第一个初始化的聚类中心
    选择出其余的聚类中心：
    计算样本中的每一个样本点与已经初始化的聚类中心之间的距离，并选择其中最短的距离
    以概率选择距离最大的样本作为新的聚类中心，重复上述过程，直到 个聚类中心都被确定
    对k个初始化的聚类中心，利用K-Means算法计算最终的聚类中心。
    """
    #f_center = KMeanspp.get_centroids(data, k)
    f_center, subCenter = KMeans.kmeans(data, k, f_center)
    # 转为数组
    f_center =f_center.getA()
    subCenter=subCenter.getA()
    center = []
    pic_new = image.new("RGB", (m, n))
    # 转换为元组，RGB形式，方便填充颜色
    for line in f_center:
        tmp = []
        for x in line:
            tmp.append(int(float(x) * 256))
        center.append(tuple(tmp))
    for i in range(len(subCenter)):
        index = subCenter[i][0]
        index_n = int(index)
        pic_new.putpixel((int(i/n),int(i % n)),tuple(center[index_n]))
        i = i + 1
    KMeans.save_result("center", f_center)
    KMeans.save_result("sub", subCenter)
    print("len",len(subCenter))
    pic_new.show()
    pic_new.save(r"./result/trump.jpg", "JPEG")
