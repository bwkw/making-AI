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
df_titanic = sns.load_dataset("titanic")

#項目名の日本語化
columns_t = ['生存', '等室', '性別', '年齢', '兄弟配偶者数', 
             '両親子供数', '料金', '乗船港コード', '等室名', 
             '男女子供', '成人男子', 'デッキ', '乗船港', '生存可否', '独身']
df_titanic.columns = columns_t

#データの内容
#display(df_titanic.head())

#欠損値の確認
#print(df_titanic.isnull().sum())

#項目「乗船港」の項目値ごとの個数
#print(df_titanic["乗船港"].value_counts())

#項目「生存可否」の項目値ごとの個数
#print(df_titanic["生存可否"].value_counts())

#統計情報の調査
#display(df_titanic.describe())

#データを集約する関数の利用
#display(df_titanic.groupby('性別').mean())


"""
分析対象項目のグラフ表示(数値項目の場合)
"""
#数値項目の定義
columns_n = ['生存', '等室', '年齢', '兄弟配偶者数', '両親子供数', '料金']

#figsizeの指定
plt.rcParams['figure.figsize'] = (8, 8)

#データフレームの数値項目でヒストグラム表示
#df_titanic[columns_n].hist()
#plt.show()


"""
分析対象項目のグラフ表示(非数値項目の場合)
"""
#グラフ化対象列の定義
columns_c = ['性別', '乗船港', '等室名', '成人男子']

#グラフ描画領域の調整
plt.rcParams['figure.figsize'] = (8, 8)

#ループ処理で、ヒストグラムの表示
for i, name in enumerate(columns_c):
    ax = plt.subplot(2, 2, i+1)
    df_titanic[name].value_counts().plot(kind='bar', title=name, ax=ax)
    
#レイアウトの調整    
#plt.tight_layout() 
#plt.show()


"""
データ前処理
"""
#余分な列を削除する
#「等室名」・「乗船港」の削除
df1 = df_titanic.drop("等室名", axis=1)
df2 = df1.drop("乗船港", axis=1)
df3 = df2.drop("生存可否", axis=1)
#display(df3.head())

#欠損値確認
#display(df3.isnull().sum())

#デッキの欠損値が多いようなので、デッキの内容を確認する
#display(df3["デッキ"].value_counts())

#乗船コードの欠損値は2件で少ない
#行ごとに削除する
#dropna関数を利用する
df4 = df3.dropna(subset=["乗船港コード"])

#年齢の欠損値は177行と多い
#他データの平均値で代用
#平均値の計算
age_average = df4["年齢"].mean()
#fillna関数の利用
df5 = df4.fillna({"年齢": age_average})

#デッキ：ラベル値データであり欠損行数が688行と相当多い
#欠損を意味するダミーコードを振って全行を処理対象とする
#replace関数の利用(ダミーコードは"N"とする)
df6 = df5.replace({"デッキ":{np.nan:'N'}})

#結果確認(乗船コードの欠損値消去・年齢の欠損値は平均値代入・デッキの欠損値をダミーコードN)
#display(df6.isnull().sum())
#display(df6.head())


"""
2値ラベルの数値化
"""
#辞書mf_mapの定義
mf_map = {"male":1, "female":0}

#map関数を利用して数値化
df7 = df6.copy()
df7["性別"] = df7["性別"].map(mf_map)

tf_map = {True:1, False:0}

df8 = df7.copy()
df8["成人男子"] = df8["成人男子"].map(tf_map)

df9 = df8.copy()
df9["独身"] = df9["独身"].map(tf_map)

#結果確認(性別・成人男子・独身を数値化)
#display(df9.head())


"""
多値ラベルの数値化
"""
#get_dummies関数の利用
w = pd.get_dummies(df9["男女子供"], prefix="男女子供")
#display(w.head())

#one-hotエンコーディングで列を追加しつつ、本のデータを削除する関数を作る
def enc(df, column):
    #One Hot Vector生成
    df_dummy = pd.get_dummies(df9[column], prefix=column)
    #元列の削除
    df_drop = df.drop(column, axis=1)
    #削除したデータフレームとOne Hot生成列を連結させる
    df1 = pd.concat([df_dummy, df_drop], axis=1)
    return df1

#項目値の確認
#display(df9["男女子供"].value_counts())

df10 = enc(df9, "男女子供")
df11 = enc(df10, "デッキ")
df12 = enc(df11, "乗船港コード")
#display(df12.head)


"""
正規化
"""
#normalizationとstandardizationの二つの方式
#外れ値を含んでいる可能性があるときは、standardization
df13 = df12.copy()
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
df13[["年齢", "料金"]] = stdsc.fit_transform(df13[["年齢", "料金"]])

#結果確認
display(df13.head())