#余分なワーニングを非表示にする
import warnings
warnings.filterwarnings("ignore")




#ライブラリのimport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#がん疾患データセットのダウンロード

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()


print(cancer.feature_names)

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

display(df[20:25])


