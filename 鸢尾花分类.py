'''
任务目的：给模型看花瓣、花萼的 4 个测量数据，训练模型学会判断这朵鸢尾花属于 3 个品种中的哪一类。

准确率：看模型整体一共预测对多少样本，简单直观；缺点是遇到类别不均衡时会产生假象。
宏平均召回率 (macro recall)：每个类别同等权重，检查会不会有某一类被模型完全识别不出来，弥补准确率的缺陷。
宏平均 F1：结合精确率和召回率，同等权衡每一类的表现，作为综合得分，用来对比挑选最好的模型。
'''


# 导入数据集、数据集划分、标准化工具
# 从sklearn的datasets模块导入load_iris函数（用于加载鸢尾花数据集）
from sklearn.datasets import load_iris
# 从sklearn的model_selection模块导入train_test_split函数（用于划分训练集和测试集）
from sklearn.model_selection import train_test_split
# 从sklearn的preprocessing模块导入StandardScaler类（用于数据标准化）
from sklearn.preprocessing import StandardScaler

# 导入4个分类算法
# 从sklearn的neighbors模块导入KNeighborsClassifier类（K近邻分类器）
from sklearn.neighbors import KNeighborsClassifier
# 从sklearn的linear_model模块导入LogisticRegression类（逻辑回归分类器）
from sklearn.linear_model import LogisticRegression
# 从sklearn的tree模块导入DecisionTreeClassifier类（决策树分类器）
from sklearn.tree import DecisionTreeClassifier
# 从sklearn的ensemble模块导入RandomForestClassifier类（随机森林分类器）
from sklearn.ensemble import RandomForestClassifier

# 从sklearn的metrics模块导入4个评估指标函数
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# 绘图库
# 导入matplotlib的pyplot模块并命名为plt（用于绘图）
import matplotlib.pyplot as plt
# 导入seaborn库并命名为sns（用于绘制统计图表，如热力图）
import seaborn as sns


# 设置matplotlib的全局字体为黑体（SimHei），解决中文显示乱码问题
plt.rcParams["font.family"] = ["SimHei"]
# 设置坐标轴负号正常显示（不显示为方块）
plt.rcParams["axes.unicode_minus"] = False


# ===== 加载数据 =====
# X,y：原始全部特征和标签
# 调用load_iris()函数，返回一个包含数据和标签的Bunch对象
iris = load_iris()
# 提取特征矩阵：150行4列，每行是一朵花的4个测量值
X = iris.data
# 提取标签向量：150个元素，每个是0/1/2代表花的种类
y = iris.target
# 提取类别名称数组：['setosa', 'versicolor', 'virginica'] (0,1,2)
class_names = iris.target_names


# ===== 划分训练集和测试集 =====
# test_size=0.2 表示测试集占20%
# random_state=42 固定随机种子，保证每次运行划分结果相同
# stratify=y 按y的类别比例分层抽样，保证训练集和测试集各类别比例一致
# 顺序固定：X_train 训练集特征   X_test 测试集特征   y_train 训练集标签   y_test 测试集标签
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===== 数据标准化 =====
# 实例化
scaler = StandardScaler()
# fit_transform：先计算训练集的均值和标准差(fit)，再用这些参数转换训练集(transform)
# 标准化公式：(x - 均值) / 标准差
X_train = scaler.fit_transform(X_train)
# transform：用训练集得到的均值和标准差转换测试集（不重新计算，避免数据泄露）
X_test = scaler.transform(X_test)


# ===== 定义模型字典 =====
# 创建一个字典，键是模型名称（字符串），值是对应的模型实例
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),  # K近邻分类器，取最近的5个邻居投票
    "逻辑回归": LogisticRegression(max_iter=200),  # 逻辑回归，最大迭代次数200次（保证收敛）
    "决策树": DecisionTreeClassifier(random_state=42),  # 决策树分类器，固定随机种子
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),  # 随机森林，包含100棵决策树
}


# ===== 训练和评估所有模型 =====
# 初始化最佳模型名称为None（空值）
best_model_name = None
# 初始化最佳F1值为-1（F1值范围0-1，-1保证第一个模型一定能更新）
best_f1 = -1
# 初始化最佳模型对象为None
best_model = None
# 创建空列表，用于保存所有模型的评估结果
results = []

# 遍历模型字典，items()取出字典，同时拿到字典的模型名和模型对象
for name, model in models.items():
    # 用训练集数据训练模型：模型学习特征和标签之间的关系
    model.fit(X_train, y_train)
    # 用训练好的模型对测试集进行预测，返回预测标签数组
    y_pred = model.predict(X_test)

    # 计算准确率：预测正确的样本数 / 总样本数
    acc = accuracy_score(y_test, y_pred)
    # 计算宏平均召回率：每个类别召回率的算术平均（不考虑各类别样本数量）
    recall = recall_score(y_test, y_pred, average="macro")
    # 计算宏平均F1值：精确率和召回率的调和平均的宏平均
    f1 = f1_score(y_test, y_pred, average="macro")

    # 将当前模型的名称和3个指标组成元组，添加到结果列表中
    results.append((name, acc, recall, f1))

    # 打印模型名称分隔线
    print(f"========{name}=========")
    # 打印准确率，:.4f表示保留4位小数
    print(f"准确率accuracy:{acc:.4f}")
    # 打印宏平均召回率
    print(f"召回率recall(macro):{recall:.4f}")
    # 打印宏平均F1值，末尾加换行符
    print(f"F1值f1(macro):{f1:.4f}\n")

    # 自动选择F1值最高的模型作为最佳模型
    # 如果当前模型的F1值大于已记录的最佳F1值
    if f1 > best_f1:
        # 更新最佳F1值为当前模型的F1值
        best_f1 = f1
        # 更新最佳模型名称
        best_model_name = name
        # 保存当前模型对象为最佳模型
        best_model = model

# ===== 用最佳模型绘制混淆矩阵 =====
# 打印最佳模型信息
print(f"最佳模型: {best_model_name} (F1={best_f1:.4f})")
# 用最佳模型对测试集进行预测（其实循环里已经预测过，这里为了代码清晰重新预测一次）
y_pred_best = best_model.predict(X_test)
# 计算混淆矩阵：3x3矩阵，行是真实标签，列是预测标签，对角线是预测正确的数量
cm = confusion_matrix(y_test, y_pred_best)

# 创建一个宽6英寸、高5英寸的画布
plt.figure(figsize=(6, 5))
# 用seaborn绘制热力图展示混淆矩阵
# annot=True 表示在每个格子里显示数值
# fmt="d" 表示数值以整数格式显示（d=decimal）
# cmap="Blues" 表示使用蓝色系配色
# xticklabels和yticklabels设置坐标轴刻度为类别名称
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
# 设置x轴标签为"预测标签"
plt.xlabel("预测标签")
# 设置y轴标签为"真实标签"
plt.ylabel("真实标签")
# 设置图表标题，包含最佳模型名称
plt.title(f"{best_model_name}-混淆矩阵")
# 自动调整子图布局，防止标签被截断
plt.tight_layout()
# 显示绘制好的图表
plt.show()