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


#分析対象項目のグラフ表示(数値項目の場合)
#数値項目の定義
columns_n = ['生存', '等室', '年齢', '兄弟配偶者数', '両親子供数', '料金']

plt.rcParams['figure.figsize'] = (8, 8)

#データフレームの数値項目でヒストグラム表示
df_titanic[columns_n].hist()
plt.show()


#分析対象項目のグラフ表示(非数値項目の場合)
#グラフ化対象列の定義
columns_c = ['性別', '乗船港', '等室名', '成人男子']

#グラフ描画領域の調整
plt.rcParams['figure.figsize'] = (8, 8)

#ループ処理で、ヒストグラムの表示
for i, name in enumerate(columns_c):
    ax = plt.subplot(2, 2, i+1)
    df_titanic[name].value_counts().plot(kind='bar', title=name, ax=ax)
    
#レイアウトの調整    
plt.tight_layout() 
plt.show()