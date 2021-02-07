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
#numpyaの浮動小数点の表示精度
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = "{:4f}".format
pd.set_option("display.max_columns", None)
plt.rcParams["font.size"] = 14
random_seed = 123


#がん疾患データセットのダウンロード

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer.feature_names)

#データフレームへの取り込み
columns = [
    "半径_平均", "きめ_平均", "周長_平均", "面積_平均",
    "平滑度_平均", "コンパクト度_平均", "凹面_平均",
    "凹点_平均", "対称性_平均", "フラクタル度_平均",
    "半径_標準誤差", "きめ_標準誤差", "周長_標準誤差",
    "面積_標準誤差", "平滑度_標準誤差" ,
    "コンパクト度_標準誤差", "凹面_標準誤差", "凹点_標準誤差",
    "対称性_標準誤差", "フラクタル度_標準誤差",
    "半径_最大", "きめ_最大", "周長_最大", "面積_最大",
    "平滑度_最大", "コンパクト度_最大", "凹面_最大", "凹点最大",
    "対称性_最大", "フラクタル度_最大"
]

df = pd.DataFrame(cancer.data, columns=columns)

y = pd.Series(cancer.target)

#入力データの行数、列数の確認
#print(df.shape)
#正解データ1と0の個数確認
#print(y.value_counts())

#散布図描画のか準備
#正解データの1と0を分割する
df0 = df[y==0]
df1 = df[y==1]
#display (df0.head())
#display (df1.head())

#散布図表示
plt.figure(figsize=(6,6))
plt.scatter(df0["半径_平均"], df0["きめ_平均"], marker="x", c="b", label="悪性")
plt.scatter(df1["半径_平均"], df1["きめ_平均"], marker="s", c="k", label="良性")
plt.grid()
plt.xlabel("半径_平均")
plt.ylabel("きめ_平均")
plt.legend()
#plt.show()

#入力データを2項目だけに絞り込む
input_columns = ["半径_平均", "きめ_平均"]
x = df[input_columns]
#display(x.head())

#訓練データと検証データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=random_seed)

#アルフゴリズム選択
algorithm = LogisticRegression(random_state=random_seed)

#学習
algorithm.fit(x_train, y_train)

#予測
y_pred = algorithm.predict(x_test)

#正解データと予測結果の比較
#正解データ 先頭から10個
y_test10 = y_test[:10].values
print(y_test10)

#予測結果 先頭から10個
y_pred10 = y_pred[:10]
print(y_pred10)