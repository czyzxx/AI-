import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
#标准化、标签编码（文字转数字）
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,classification_report

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

# 数据集：891条乘客数据，Survived=1存活，0遇难
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
print("原始数据集大小：",df.shape)
# 查看基础信息
print(df.info())
print(df.describe())

# 删除无关列
df=df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
# 缺失值填充
df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
# 类别特征编码
le_sex = LabelEncoder() #实例化一个标签编码器对象
df["Sex"] = le_sex.fit_transform(df["Sex"]) #扫描这列文本，收集所有不同类类别并映射成数字
le_embark = LabelEncoder()
df["Embarked"] = le_embark.fit_transform(df["Embarked"])

# 存活人数分布
plt.figure(figsize=(5,4))
sns.countplot(x="Survived",data=df)
plt.title("存活与遇难人数分布")
plt.show()
# 性别与存活关系
plt.figure(figsize=(5,4))
sns.countplot(x="Sex",hue="Survived",data=df)
plt.title("性别-存活分布")
plt.show()
# 票价分布
plt.figure(figsize=(6,4))
sns.histplot(df["Fare"],bins=20) #20个小区间
plt.title("乘客票价分布")
plt.show()

# 用特征X预测标签y
X=df.drop("Survived",axis=1)
y=df["Survived"]
# 8:2划分，固定随机种子保证复现
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# KNN需要特征标准化
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


models={
    "KNN":KNeighborsClassifier(n_neighbors=5),
    "逻辑回归":LogisticRegression(max_iter=300),
    "决策树":DecisionTreeClassifier(random_state=42),
    "随机森林":RandomForestClassifier(n_estimators=100,random_state=42)
}

result_list=[]


for name,model in models.items():
    print(f"\n====== {name} ======")
    # KNN用标准化数据，其余模型用原始数据
    if name=="KNN":
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
    else:
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

    # 计算指标：准确率、召回率、F1
    acc=accuracy_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    print(f"准确率:{acc:.4f} 召回率:{rec:.4f} F1:{f1:.4f}")
    print("分类报告：")
    print(classification_report(y_test,y_pred))

    # 混淆矩阵
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues") #热力图，annot=True格子里显示数字,fmt="d"格式为整数 decimal，cmap="Blues"：配色用蓝色，数值越大颜色越深
    plt.title(f"{name} 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

    result_list.append({"模型":name,"准确率":acc,"召回率":rec,"F1":f1})


res_df=pd.DataFrame(result_list)
print("\n=====全部模型性能汇总=====")
print(res_df)

# ==========9.结果分析（写报告直接复制这段话） ==========
'''
1.随机森林综合指标最高：集成多棵决策树，抗过拟合，捕捉特征非线性关系；
2.逻辑回归表现次之：线性模型，可解释性强，适合二分类任务；
3.决策树容易过拟合测试集，泛化能力弱；
4.KNN依赖特征尺度，必须做标准化；对噪声数据比较敏感。

> 不能只看准确率：本数据集样本不平衡，部分遇难样本多，召回率、F1更加反映模型识别存活乘客的能力。
'''