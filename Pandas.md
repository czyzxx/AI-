# Pandas
Pandas是Python数据分析工具链中最核心的库，充当数据读取、清洗、分析、统计、输出的高效工具。

Pandas提供了易于使用的数据结构和数据分析工具，特别适合于处理结构化数据，如表格型数据。

Pandas是数据科学和分析领域中常用的工具之一，它使得用户能轻松地从各种数据源中导入数据，并对数据进行高效的操作和分析。
![](D:\pandas.png)
![](D:\pandas1.png)

## Series
![](D:\series.png)

### 创建方法
#### series的创建
````python
import pandas as pd
s=pd.Series([1,2,3,4,5])
print(s)
"""
0    1
1    2
2    3
3    4
4    5
"""
````
#### 自定义索引
````python
s=pd.Series([10,2,3,4,5],index=['A','B','C','D','E'])
print(s)
"""
A    10
B    2
C    3
D    4
E    5
"""
````
#### 通过字典来创建
````python
s=pd.Series({"a":1,"b":2,"c":3,"d":4})
print(s)
s1=pd,Series(s,index=["a","c"])
print(s1)
"""
a    1
b    2
c    3
d    4
dtype:int64
a    1
c    3
dtype:int64
"""
````

### Series的属性
![](D:\series属性.png)
````python
print(s.index) #Index(['a','b','c','d'],dtype='object')
print(s.values) #[1 2 3 4 5]
print(s.shape,s.ndim,s.size) #(5,) 1 5
s.name='test'
print(s.dtype,s.name) #int64 test
print(s.loc['a']) #显式索引 1
print(s.loc['a':'c']) #左右都取
"""
a    1
b    2
c    3
"""
print(s.iloc[0]) #隐式索引 1
print(s.iloc[0:2]) #左闭右开
"""
a    1
b    2
"""
print(s.at['a']) #1 不支持切片
print(s.iat[0]) #1 不支持切片
````

### 访问数据
````python
print(s['a']) #1
print(s[s<3])
"""
a    1
b    2
"""
s['e']=5
s['f']=6
print(s.head()) #只打印前五行
"""
a    1
b    2
c    3
d    4
e    5
"""
print(s.tail()) #只打印后五行
"""
b    2
c    3
d    4
e    5
f    6
"""
print(s.tail(1)) #f    6
````

### Series的常用方法
![](D:\series常用方法.png)
![](D:\series常用方法2.png)
````python
s=pd.Series([10,2,np.nan,None,3,4,5],index=['a','b','c','d','e','f','g'],name='data') #np.nan为浮点数类型
print(s)
"""
a    10.0
b    2.0
c    NaN
d    NaN
e    3.0
f    4.0
g    5.0
Name=data
"""
print(s.head())
print(s.tail())
````
#### 查看所有描述性信息
````python
print(s.describe()) #不计算缺失值
"""
data
count    5.000000
mean     4.800000
std      3.114482
min      2.000000
25%      3.000000
50%      4.000000
75%      5.000000
max      10.000000
"""
````
#### 获取元素个数
````python
print(s.count()) #5 不计算缺失值
````
#### 获取索引
````python
print(s.keys()) #Index(['a','b','c','d','e','f','g'],dtype='object')  方法
print(s.index) #Index(['a','b','c','d','e','f','g'],dtype='object')  属性
````
#### 缺失值
````python
print(s.isna()) #检查series里的每一个元素是否为缺失值
"""
a    False
b    False
c    True
d    True
e    False
f    False
g    False
"""
````
#### 
````python
print
````
## DataFrame
![](D:\dataframe的属性.png)
## 数据分析的完整流程
数据收集--数据清洗--数据分析--数据可视化