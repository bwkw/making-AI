#余分なワーニングを非表示にする
import warnings
warnings.filterwarnings("ignore")


#ライブラリのimport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#表示オプションの調整
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = "{:4f}".format
pd.set_option("display.max_columns", None)
plt.rcParams["font.size"] = 10
random_seed = 123


#ライブラリimport
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification


#線形分離型
X1, y1 = make_classification(n_features=2, n_redundant=0,
    n_informative=2, random_state=random_seed,
    n_clusters_per_class=1, n_samples=200, n_classes=2)


#三日月型(線形分離不可)
X2, y2 = make_moons(noise=0.05, random_state=random_seed,
    n_samples=200)


#円形(線形分離不可)
X3, y3 = make_circles(noise=0.02, random_state=random_seed,
    n_samples=200)


#3種類のデータをDataListに保存
DataList = [(X1, y1), (X2, y2), (X3, y3)]


#N:データの種類
N = len(DataList)


"""
散布図表示
"""
plt.figure(figsize=(15,4))

#カラーマップ定義
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#0000FF', '#000000'])

for i, data in enumerate(DataList):
    X, y = data
    ax = plt.subplot(1, N, i+1)
    #[:, 0]は全ての行の0列目
    ax.scatter(X[:, 0], X[:,1], c=y, cmap=cmap)

plt.show()


"""
ロジスティック回帰
"""
#シグモイド関数の定義
#値が単調に増え続けている
#値が0から1の間をとっている
#グラフは点対象な形になっていて、対象の中心はx=0, y=0.5の点

def sigmoid(x):
    return 1/(1+np.exp(-x))

#xのデータ準備
x = np.linspace(-5, 5, 101)

#yのデータ準備
y = sigmoid(x)

#グラフ表示
plt.plot(x, y, label="シグモイド関数", c="b", lw=2)

#凡例表示
plt.legend()

#方眼表示
plt.grid()

#グラフ描画
plt.show()

#アルゴリズムの選択
from sklearn.linear_model import LogisticRegression
algorithm = LogisticRegression(random_state=random_seed)

#アルゴリズムの持つパラメータの表示
print(algorithm)

from sklearn.model_selection import train_test_split

# 決定境界の表示関数
def plot_boundary(ax, x, y, algorithm):
    x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.5, random_state=random_seed)
    # カラーマップ定義
    from matplotlib.colors import ListedColormap
    cmap1 = plt.cm.bwr
    cmap2 = ListedColormap(['#0000FF', '#000000'])

    h = 0.005
    algorithm.fit(x_train, y_train)
    score_test = algorithm.score(x_test, y_test)
    score_train = algorithm.score(x_train, y_train)
    f1_min = x[:, 0].min() - 0.5
    f1_max = x[:, 0].max() + 0.5
    f2_min = x[:, 1].min() - 0.5
    f2_max = x[:, 1].max() + 0.5
    f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, h), 
                         np.arange(f2_min, f2_max, h))
    if hasattr(algorithm, "decision_function"):
        Z = algorithm.decision_function(np.c_[f1.ravel(), f2.ravel()])
        Z = Z.reshape(f1.shape)
        ax.contour(f1, f2, Z, levels=[0], linewidth=2)
    else:
        Z = algorithm.predict_proba(np.c_[f1.ravel(), f2.ravel()])[:, 1]
        Z = Z.reshape(f1.shape)
    ax.contourf(f1, f2, Z, cmap=cmap1, alpha=0.3)
    ax.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cmap2)
    ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cmap2, marker='x')
    text = f'検証:{score_test:.2f}  訓練: {score_train:.2f}'
    ax.text(f1.max() - 0.3, f2.min() + 0.3, text, horizontalalignment='right',
    fontsize=18)
    
#散布図と決定境界の表示関数
def plot_boundaries(algorithm, DataList):
    plt.figure(figsize=(15,4))
    for i, data in enumerate(DataList):
        X, y = data
        ax = plt.subplot(1, N, i+1)
        plot_boundary(ax, X, y, algorithm)
    plt.show()

#表示関数の呼び出し
plot_boundaries(algorithm, DataList)

