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


#追加ライブラリのimport
import seaborn as sns


#サンプルデータの読み込み
df_iris = sns.load_dataset("iris")


#項目名の日本語化
df_iris.columns = ["がく片長", "花弁長", "花弁幅", "種別"]


#データの内容
display(df_iris.head())


#matplotlibを用いた散布図表示
#グラフ領域の調整
plt.figure(figsize=(6, 6))

#散布図の表示
plt.scatter(df_iris["がく片幅"], df_iris["花弁長"])

#ラベル表示
plt.xlabel("がく片幅")
plt.ylabel("花弁長")
plt.show()


#seabornを用いた散布図表示
plt.figure(figsize=(6, 6))
sns.scatterplot(x="がく片幅", y="花弁長", hue="種別", s=70, data=df_iris)
plt.show()


#全散布図同時表示
sns.pairplot(df_iris, hue="種別")
plt.show()


#散布図表示
sns.jointplot("がく片幅", "花弁長", data=df_iris)
plt.show()


#matplotlibを用いた箱髭図表示
#グラフ描画領域の調整
plt.figure(figsize=(6, 6))

#箱髭図の描画
#patch_artistは長方形を塗りつぶすために用いられたもの
df_iris.boxplot(patch_artist=True)
plt.show()


#seabornを用いた箱髭図表示
#melt関数によるデータの事前加工
w = pd.melt(df_iris, id_vars=["種別"])

#加工結果の確認
display(w.head())

#hueパラメータを追加し、花の種類で箱髭図を書き分ける
plt.figure(figsize=(8, 8))
sns.boxplot(x="variable", y="value", data=w, hue="種別")
plt.show()