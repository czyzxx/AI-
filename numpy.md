# numpy
## 一、ndarray的特性
多维性、同质性、高效性

## 二、ndarray的属性
arr.shape    数组的形状（行、列等）

arr.ndim     维度数量（1、2、3维等）

arr.size     总元素个数（数组中所有元素的总数）

arr.dtype    元素类型（整数、浮点数等）

arr.T        转置（行变列，列变行）

arr.itemsize 单个元素占用的内存字节数

arr.nbytes   数组总内存占用量（size*itemsize）

arr.flags    内存存储方式（是否连续存储）

## 三、ndadday的创建
![](D:\微信图片_20260807155814.png)
### 1.基础的创建方法
````python
import numpy as np
list1=[4,5,6]
arr=np.array(list1,dtype=np.float64)
print(arr.ndim)
print(arr)
````
#### copy
````python
arr1=np.copy(arr)
print(arr1)
arr1[0]=8
print(arr1)
print(arr)
````
### 2.预定义形状
#### 全0
````python
arr=np.zeros((2,3))
print(arr) 
#运行结果
#[[0. 0. 0.]
# [0. 0. 0.]]
````
#### 全1
````python
arr=np.zeros((2,3),dtype=int)
print(arr) 
#运行结果
#[[1 1 1]
# [1 1 1]]
````
#### 未初始化
````python
arr=np.emptty((2,3))
print(arr) 
#运行结果不确定
````
#### 固定值
````python
arr=np.full((3,4),2025)
print(arr)
#[[2025 2025 2025 2025]
# [2025 2025 2025 2025]
# [2025 2025 2025 2025]]
arr1=np.zeros_like(arr)
print(arr1)
#[[0 0 0 0]
# [0 0 0 0]
# [0 0 0 0]]
````
### 3.基于数值范围
#### 等差数列
````python
arr=np.arange(1,10,1) #start,end,step
print(arr)
#[1 2 3 4 5 6 7 8 9]
````
#### 等间隔数列
````python
arr=np.linspace(1,10,5)
print(arr)
#[1. 3.25 5.5 7.75 10.]
````
#### 对数间隔数列
````python
arr=np.logspace(0,4,3,base=2) #base默认为10
print(arr)
#[1. 4. 16.] 2的0 2的2 2的4
````
### 4.特殊矩阵
![](D:\矩阵.png)
#### 单位矩阵
````python
arr=np.eye(3,dtype=int)
print(arr)
#[[1 0 0]
# [0 1 0]
# [0 0 1]]
````
````python
arr=np.eye(3,4,dtype=int)
print(arr)
#[[1 0 0 0]
# [0 1 0 0]
# [0 0 1 0]]
````
#### 对角矩阵
````python
arr=np.diag([1,2,3])
print(arr)
#[[1 0 0]
# [0 2 0]
# [0 0 3]]
````
### 5.随机数组的生成
#### 生成0-1之间的随机浮点数（均匀分布）
````python
arr=np.random.rand(2,3) #2行3列
print(arr)
````
#### 生成指定范围区间的随机浮点数
````python
arr=np.random.uniform(3,6,(2,3))
print(arr)
````
#### 生成指定范围区间的随机整数
````python
arr=np.random.randint(3,30,(2,3))
print(arr)
````
#### 生成随机数列（正态分布）
两边的概率小，中间的概率大（-3，3）
````python
arr=np.random.randn(2,3)
print(arr)
````
#### 设置随机种子
````python
np.random.seed(20)
arr=np.random.randint(1,10,(2,5))
print(arr)
#每次生成结果均相同
````
## 四、ndarray的数据类型
![](D:\数据类型.png)
````python
arr=np.array([1,0,2,0],dtype='bool')
print(arr)
#[True False True False]
````
````python
arr=np.array([1,0,127,0],dtype=np.int8)
print(arr)
#[1 0 127 0]
````
## 五、索引与切片
![](D:\索引与切片.png)
### 1、一维数组的索引与切片
````python
arr=np.random.randint(1,100,20)
print(arr)
print(arr[0])
print(arr[:]) #获取全部数据
print(arr[2:5]) #获取下标为2到4的元素，左包右不包
print(arr[arr>10]) #布尔索引
print(arr[(arr>10)&(arr<70)])
print(arr[slice(2,5)]) #获取下标为2到4的元素
print(arr[slice(2,15,3)]) #start,end,step
````
### 2、二维数组的索引与切片
````python
arr=np.random.randint(1,100,(4,8))
print(arr)
print(arr[1][3])
print(arr[:,:]) #获取全部数据
print(arr[1,:]) #获取第一行所有数据
print(arr[1,2:5]) #获取第一行下标为2到4的元素，左包右不包
print(arr[arr>50]) #布尔索引，返回一维数组
print(arr[2][arr[2]>50]) #第二行大于50的数
print('每列的最大值',np.max(arr,axis=0)) #axis=0列 axis=1行
print(arr[(arr>10)&(arr<70)])
````
## 六、ndarray的运算
### 1.算术运算
````python
a=np.array([1,2,3])
b=np.array([4,5,6])
print(a+b) #+,-,*,/,**
#[5,7,9]
c=[1,2,3]
d=[4,5,6]
print(c+d)
#[1,2,3,4,5,6]
for i in range(len(c)):
    d[i]=d[i]+c[i]
print(d)
#[5,7,9]
````
### 2.广播机制
同一维度：相同、1
````python
a=np.array([1,2,3])
b=np.array([4],[5],[6])
print(a+b) 
#[[5 6 7]
# [6 7 8]
# [7 8 9]]
````
### 3.矩阵运算
````python
a=np.array([1,2,3],[4,5,6],[7,8,9])
b=np.array([4,5,6],[7,8,9],[1,2,3])
print(a*b) 
#[[4 10 18]
# [28 40 54]
# [7 16 27]
print(a@b) #@线性代数矩阵的乘法
#[[21 27 33]
# [57 72 87]
# [93 117 141]]
````
## 七、numpy常用函数
![](D:\numpy常见函数.png)
### 1.基本数学函数
#### 计算平方根
````python
print(np.sqrt(9)) #3.0
print(np.sqrt([1,4,9])) #[1. 2. 3.]
arr=np.array([1,25,81]) 
print(np.sqrt(arr)) #[1. 5. 9.]
````
#### 计算指数
默认以e为底，e^x
````python
print(np.exp(0)) #1.0 
````
#### 计算自然对数
lnx
````python
print(np.log(np.e))  #1.0
````
#### 计算正弦值、余弦值
````python
print(np.sin(-1)) 
print(np.cos(np.pi)) #-1.0  pi=Π
````
#### 计算绝对值
````python
arr=np.array([-1,1,2,-3])
print(np.abs(arr)) 
````
#### 计算a的b次幂
````python
arr=np.array([-1,1,2,-3])
print(np.power(arr,2))  #[1 1 4 9]
````
#### 四舍五入
````python
print(np.round([3.2,4.5,8.51,9.6]))  #[3. 4. 9. 10.]
````
#### 向上取整，向下取整
````python
arr=np.array([1.6,25.1])
print(np.ceil(arr)) #[2. 26.] 
print(np.floor(arr))  #[1. 25.]
````
#### 检测缺失值NaN
````python
np.isnan([1,2,np.nan]) #array([False,False,True])
````
### 2.统计函数
````python
arr=np.random.randint(1,20,8)
print(arr)
````
#### 求和
````python
print(np.sum(arr))
print(np.sum([1,2,3])) #6
````
#### 计算平均值
````python
print(np.mean(arr))
print(np.mean([1,2,3])) #2.0
````
#### 计算中位数
````python
print(np.median(arr))
````
#### 计算标准差、方差
方差：

计算平均值

每个数与平均值差的平方相加

除以个数

标准差：

在第二步的基础上开根号
````python
print(np.var([1,2,3])) #方差
print(np.std([1,2,3])) #标准差
````
#### 计算最大值最小值
````python
print(np.max(arr),np.argmax(arr))#arg索引
print(np.min(arr),np.argmin(arr)) 
````
#### 分位数
````python
print(np.median(arr))
print(np.percentile(arr,25))
````
#### 累积和、累积积
````python
print(np.sum(arr))
print(np.cumsum([1,2,3])) #[1 3 6] 
print(np.cumprod([1,2,3])) #[1,2,6]
````
### 3.比较函数
#### 是否大于、小于、等于
````python
print(np.greater([3,4,5,6],4)) #[False False True True]
print(np.less([3,4,5,6],4)) #[True False False False]
print(np.equal([3,4,5,6],4)) #[False True False False]
````
#### 逻辑
````python
print(np.logical_and([1,0],[1,1])) #[True False]
print(np.logical_not([1,0])) #[False True]
print(np.logical_or([0,0],[1,0])) #[True False]
````
#### 检查元素是否至少有一个元素为True
````python
print(np.any([1,0])) #True
print(np.any([0,0,0])) #False
````
#### 自定义条件
````python
#print(np.where(条件，符合条件，不符合条件的))
arr=np.array([1,2,3,4,5])
print(np.where(arr<3,arr,0)) #[1 2 0 0 0]
````
### 4.排序函数
````python
np.randon.seed(0)
arr=np.random.randint(1,100,20)
````
#### 从小到大
````python
print(arr.sort()) #None
print(arr) #从小到大
````
````python
print(np.sort(arr)) #从小到大
print(np.argsort(arr)) #索引，在新排序数组中每个元素在原数组的位置
print(arr) #原数组
````
### 5.去重函数
````python
print(no.unique(arr))
````
### 6.数组的拼接
````python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2) #[5 7 9]
print(np.concatenate((arr1,arr2))) #[1 2 3 4 5 6]
````
### 7.数组的分割
````python
print(np.split(arr,4)) #分成了4份，不能正好等分则报错
````
### 8.调整数组的形状
````python
print(np.reshape(arr,[5,4])) #同样必须满足等分
````
