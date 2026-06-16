# 导入数据计算库
import numpy as np
# 导入表格处理库
import pandas as pd
# 导入绘图库
import matplotlib.pyplot as plt
# 从torchvision导入经典数据集工具datasets
from torchvision import datasets
# 导入操作系统文件处理模块os，用来创建文件夹
import os

# 中文显示修复
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创造名为img的文件夹，exist_ok=True
os.makedirs("img", exist_ok=True)

# 下载&加载MNIST训练集：数据存放在./data目录，train=True,download=True没有就自动联网下载
train = datasets.MNIST(root="./data", train=True, download=True)
# 把训练集图片张量转为numpy数组格式
imgs = train.data.numpy()
# 把标签张量转为numpy数组格式
labels = train.targets.numpy()

# 类别分布
# np.bincount统计0~9每个数字标签出现的样本个数
cnt = np.bincount(labels)
# pandas构建数据表：一列数字0~9，一列样本量
df = pd.DataFrame({"数字":range(10), "样本量":cnt})
# 新建一张画布
plt.figure()
# 画柱状图：x=数字，y=对应样本数量
plt.bar(df["数字"], df["样本量"])
# 设置表图标题
plt.title("各类别样本分布")
# 把柱状图保存到img文件夹，命名class_dist.png
plt.savefig("img/class_dist.png")

# 样本预览
# 新建画布，尺寸8*8
plt.figure(figsize=(8,8))
# 遍历循环25张照片
for i in range(25):
    # 划分子图：5行5列，当前是i+1个子图
    plt.subplot(5,5,i+1)
    # 绘制第i张图片，灰度图显示
    plt.imshow(imgs[i], cmap="gray")
    # 关闭坐标轴刻度、边框
    plt.axis("off")
plt.savefig("img/sample.png")

# 灰度直方图
# 把所有图片的二维像素展平成一维数组，汇总全部像素值
all_pix = imgs.ravel()
plt.figure()
# 绘制直方图：bins=50，分成50个区间统计像素灰度频次
plt.hist(all_pix, bins=50)
plt.title("像素灰度分布")
plt.savefig("img/gray_hist.png")