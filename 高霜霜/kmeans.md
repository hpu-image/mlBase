# k-means算法

## 相关概念

**聚类**：“类”指的是具有相似性的集合。聚类是指将数据集划分为若干类，使得类内之间的数据最为相似，各类之间的数据相似度差别尽可能大。聚类分析就是以相似性为基础，对数据集进行聚类划分，属于无监督学习

**无监督学习**：K-均值聚类属于无监督学习。监督学习知道从对象（数据）中学习什么，而无监督学习无需知道所要搜寻的目标，它是根据算法得到数据的共同特征。比如用分类和聚类来说，分类事先就知道所要得到的类别，而聚类则不一样，只是以相似度为基础，将对象分得不同的簇

**SSE**：手肘法的核心指标为SSE(误差平方和)，是所有样本的聚类误差，代表了聚类效果的好坏

**簇**：所有数据点点集合，簇中的对象是相似的

**质心**：簇中所有点的中心（计算所有点的均值而来）

## 算法流程

K均值算法是一个迭代算法，主要做两件事情，第一件是簇分配，第二件是移动聚类中心

1. 随机选k个点作为初代的聚类中心点
2. 将数据集中的每个点分配到一个簇中，即为每个点找距离其最近的质心，并将其分配给质心所对应的簇
3. 簇分好后，计算每个簇所有点的平均值，将平均值作为对应簇新的质心
4. 循环2~3步骤，直到簇中心不再改变或者达到指定的迭代次数

## 手肘法

计算公式：
$$
SSE=\sum_{k}^{i=1}\sum_{x_i\in C_i}|x_i-m_i|^2 \quad \quad m_i=\frac{1}{|C_i|}\sum_{x_i\in C_i}x_i
$$
核心思想：
        随着聚类数k的增大，样本划分更加精细，每个簇的聚合程度会逐渐提高，误差平方和SSE自然会逐渐变小。
        当k小于真实聚类数时，由于k的增大会大幅增加每个簇的聚合程度，故SSE的下降幅度会很大，而当k到达真实聚类数时，再增加k所得到的聚合程度回报会迅速变小，所以SSE的下降幅度会骤减，然后随着k值的继续增大而趋于平缓，也就是说 SSE和k的关系图是一个手肘的形状，而这个肘部对应的k值就是数据的真实聚类数，我们则选取肘部对应的k作为我们的最佳聚类数

![image-20201120153429124](E:\AAAAAA学习\机器学习\kmean\高霜霜 Kmeans算法\kmeans.assets\image-20201120153429124.png)

显然，肘部对应的k值为4，故对于这个数据集的聚类而言，最佳聚类数应该选4

## 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
 
# 加载数据
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter='\t')  #使用delimiter参数进行分割，默认是将整个数据一起输出
    return data
 
# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  
 
# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))   #初始化质心,创建(k,n)个以零填充的矩阵
    for i in range(k):  # 循环遍历特征值
        index = int(np.random.uniform(0,m)) #从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        centroids[i,:] = dataSet[index,:]  # 计算每一列的质心,并将值赋给centroids
    return centroids
 
# k均值聚类
def KMeans(dataSet,k):
 
    m = np.shape(dataSet)[0]  #行的数目
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:第一列存样本属于哪一簇 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True
 
    # 第1步 初始化centroids 创建质心,随机K个质心
    centroids = randCent(dataSet,k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    while clusterChange:
        clusterChange = False
 
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
 
            #第2步 遍历所有数据找出每个点最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:]) # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                 # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distance < minDist: 
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                clusterAssment[i,:] = minIndex,minDist**2
        #第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的列求均值
 
    print("Congratulations,cluster complete!")
    return centroids,clusterAssment  # 返回所有的类质心与点分配结果
 
def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1
 
    mark = ['or', 'ob', 'og', 'ok', '^b', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1
 
    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
 
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i],color = 'yellow')
 
    plt.show()
dataSet = loadDataSet("testSet.txt")
k = 4
centroids,clusterAssment = KMeans(dataSet,k)
 
showCluster(dataSet,k,centroids,clusterAssment)
```

结果：

![image-20201120153640558](E:\AAAAAA学习\机器学习\kmean\高霜霜 Kmeans算法\kmeans.assets\image-20201120153640558.png)

## 相关内容

* 每个类别中心的初始点如何选择：

  随机法：最简单的确定初始类簇中心点的方法是随机选择K个点作为初始的类簇中心点
  这k个点的距离尽可能远：首先随机选择一个点作为第一个初始类簇中心点，然后选择距离该点最远的那个点作为第二个初始类簇中心点，然后再选择距离前两个点的最近距离最大的点作为第三个初始类簇的中心点，以此类推，直到选出k个初始类簇中心。

* K-means算法的优点和缺点：

  主要优点：原理简单，容易实现、可解释度较强
  主要缺点：K值很难确定、局部最优、对噪音和异常点敏感、需样本存在均值（限定数据种类）、聚类效果依赖于聚类中心的初始化、对于非凸数据集或类别规模差异太大的数据效果不好

