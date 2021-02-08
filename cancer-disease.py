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
#print(cancer)
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


#散布図描画の準備
#正解データの1と0を分割する
df0 = df[y==0]
df1 = df[y==1]
#display (df0.head())
#display (df1.head())


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
#print(y_pred)
#print(y_test)テストはSeriesであることに注意


#予測の精度確認1
y_test_values = y_test.values
#print(y_test_values)
match = 0
for i in range(len(y_test_values)):
    if y_test_values[i]==y_pred[i]:
        match += 1
    else:
        continue
accuracy = match/len(y_test_values)
#print(accuracy)


#予測の精度確認2
score = algorithm.score(x_test, y_test)
print(f'入力  2 項目: {score:.04f}')


#一応全データで学習させてみる
x2_train, x2_test, y_train, y_test = train_test_split(df, y, 
    train_size=0.7, test_size=0.3, random_state=random_seed)
algorithm2 = LogisticRegression(random_state=random_seed)
algorithm2.fit(x2_train, y_train)
score2 = algorithm2.score(x2_test, y_test)
print(f'入力 30 項目: {score2:.04f}')


#散布図と境界表示
#内部変数値の取得
w0 = algorithm.intercept_[0]
w1 = algorithm.coef_[0,0]
w2 = algorithm.coef_[0,1]

#境界表示の関数
def boundary(algorithm, x):
    w1 = algorithm.coef_[0][0]
    w2 = algorithm.coef_[0][1]
    w0 = algorithm.intercept_[0]
    y = -(w0 + w1 * x)/w2
    return y

#決定境界の端点のx座標
x_range = np.array((df['半径_平均'].min(), df['半径_平均'].max()))

#決定境界の端点のy座標
y_range = boundary(algorithm, x_range)

#yの上限、下限は散布図の点を元に決める
y_lim = (df['きめ_平均'].min(), df['きめ_平均'].max())

# グラフのサイズ設定
plt.figure(figsize=(6,6))

# 目的変数が0のデータを散布図表示
plt.scatter(df0['半径_平均'], df0['きめ_平均'], marker='x', c='b', label='悪性')

# 目的変数が1のデータを散布図表示
plt.scatter(df1['半径_平均'], df1['きめ_平均'], marker='s', c='k', label='良性')

# 決定境界
plt.plot(x_range, y_range, c='r', label='決定境界')

# 範囲指定
plt.ylim(y_lim)

# ラベル表示
plt.xlabel('半径_平均')
plt.ylabel('きめ_平均')

# 凡例表示
plt.legend()

# 方眼表示
plt.grid()

# グラフ表示
plt.show()
