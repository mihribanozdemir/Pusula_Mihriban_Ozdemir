import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

dff = pd.read_excel("/content/side_effect_data.xlsx")

df = dff.copy()

df.info()

df.head()

"""Veri Seti İle İlgili Bilgilere Erişelim"""

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)

"""grab_col_names fonksiyonu sayesinde kategorik,numerik değişkenleri ayıklayalım"""

def grab_col_names(dataframe, cat_th=10, car_th=20):
  ###cat_cols
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                 dataframe[col].dtypes != "O"]
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                 dataframe[col].dtypes == "O"]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]
  ###num_cols
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
  num_cols = [col for col in num_cols if col not in num_but_cat]
  print(f"observations: {dataframe.shape[0]}")
  print(f"variables: {dataframe.shape[1]}")
  print(f"cat_cols: {len(cat_cols)}")
  print(f"num_cols: {len(num_cols)}")
  print(f"cat_but_car: {len(cat_but_car)}", f"cat_but_car name: {cat_but_car}")
  print(f"num_but_cat: {len(num_but_cat)}", f"num_but_cat name: {num_but_cat}")
  return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""İlacın Kullanıldığı Gün Sayısı"""

df["toplam_ilac_kullanilan_gun"] = df["Ilac_Bitis_Tarihi"] - df["Ilac_Baslangic_Tarihi"]

df["toplam_ilac_kullanilan_gun"].dt.days

df.info()

"""Yan Etkinin Başlangıç Tarihini İçin Sütun"""

df["Yan_etki_baslangic_tarihi"] = (df["Yan_Etki_Bildirim_Tarihi"] - df["Ilac_Baslangic_Tarihi"]).dt.days

"""Yaş Değişkeni"""

import datetime as dt

today_date = dt.datetime(2024, 9, 22)
df["Yaş"] = (today_date - df["Dogum_Tarihi"]).dt.days // 365
df.drop("Dogum_Tarihi", inplace=True, axis=1)

"""Eksik Verilerin Kontrolü"""

for col in df.columns:
    print(f'{col}: {df[col].isnull().sum()}')

"""Gereksiz Sütunların Silinmesi"""

df = df.drop(columns = ["Kullanici_id","Ilac_Bitis_Tarihi", "Ilac_Baslangic_Tarihi","Yan_Etki_Bildirim_Tarihi"])

df.info()

"""Summary Table"""

def summary_table(df):
    """
    Return a summary table with the descriptive statistics about the dataframe.
    """
    summary = {
    "Number of Variables": [len(df.columns)],
    "Number of Observations": [df.shape[0]],
    "Missing Cells": [df.isnull().sum().sum()],
    "Missing Cells (%)": [round(df.isnull().sum().sum() / df.shape[0] * 100, 2)],
    "Duplicated Rows": [df.duplicated().sum()],
    "Duplicated Rows (%)": [round(df.duplicated().sum() / df.shape[0] * 100, 2)],
    "Categorical Variables": [len([i for i in df.columns if df[i].dtype==object])],
    "Numerical Variables": [len([i for i in df.columns if df[i].dtype!=object])],
    }
    return pd.DataFrame(summary).T.rename(columns={0: 'Values'})

summary_table(df)

"""grab_col_names fonksiyonumuzu tekrar çalıştıralım"""

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""Kategorik Değişkenlerin Analizi"""

def cat_summary(dataframe, col_name):
  print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                      "ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
  print("##############done")


for col in cat_cols:
  cat_summary(df, col)

"""Numerik Değişkenlerin Analizi"""

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[col_name].describe(quantiles).T)
    if plot:
      dataframe[col_name].hist(bins=20)
      plt.xlabel(col_name)
      plt.title(col_name)
      plt.show(block=True)


for col in num_cols:
    num_summary(df, col)

"""toplam_ilac_kullanilan_gun sutununun veri tipini değiştirelim"""

df["toplam_ilac_kullanilan_gun"] = df["toplam_ilac_kullanilan_gun"].dt.days

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""Eksik Verilerin Görselleştirilmesi"""

import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

custom_palette = sns.light_palette("seagreen", as_cmap=True)

msno.matrix(df, color=(0.25, 0.6, 0.8))
plt.show()

msno.bar(df, color='seagreen')
plt.show()

"""Eksik Verilerin Gözlenmesi"""

def missing_values_table (dataframe, na_name:False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys = ["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df, False)

"""Dtype = object olan sütünların boş değerlerini "Bilinmiyor" İle Dolduralım"""

def fill_object_na_with_bilinmiyor(dataframe):
    """
    DataFrame'deki dtype'ı 'object' olan sütunlardaki eksik değerlere 'Bilinmiyor' atar.

    Parameters:
    dataframe: pd.DataFrame
        İşlem yapılacak veri seti.

    Returns:
    dataframe: pd.DataFrame
        Eksik değerlere 'Bilinmiyor' atanmış veri seti.
    """
    # dtype'ı object olan sütunları seç
    object_columns = dataframe.select_dtypes(include=['object']).columns

    dataframe[object_columns] = dataframe[object_columns].fillna('Bilinmiyor')

    return dataframe


df = fill_object_na_with_bilinmiyor(df)


df.isnull().sum()

"""Sayısal olup, null değerleri olanları median ile dolduralım"""

def fill_numeric_na_with_median(dataframe):
    """
    DataFrame'deki sayısal sütunlardaki eksik değerlere medyan değerlerini atar.

    Parameters:
    dataframe: pd.DataFrame
        İşlem yapılacak veri seti.

    Returns:
    dataframe: pd.DataFrame
        Sayısal sütunlardaki eksik değerleri medyan ile doldurulmuş veri seti.
    """

    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].fillna(dataframe[numeric_columns].median())

    return dataframe


df = fill_numeric_na_with_median(df)


df.isnull().sum()  # Eksik değer olup olmadığını kontrol etmek için

"""Artık veri setimizde eksik değer yok.

Görselleştirme

Numeric sutunların görselleştirilmesi
"""

for col in num_cols:
    num_summary(df, col, plot=True)

"""categoric sutunların görselleştirilmesi"""

import seaborn as sns
import matplotlib.pyplot as plt

def plot_categorical_variables(dataframe, cat_cols):
    """
    Kategorik değişkenlerin bar grafiklerini oluşturur.

    Parameters:
    dataframe: pd.DataFrame
        Veri seti.
    cat_cols: list
        Kategorik sütunların isimlerinin listesi.
    """
    for col in cat_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=dataframe[col], data=dataframe, palette='Set2')
        plt.title(f'{col} Değişkeni Dağılımı')
        plt.xticks(rotation=45)
        plt.show()


cat_cols, num_cols, cat_but_car = grab_col_names(df)


plot_categorical_variables(df, cat_cols)

df.info()

"""Yaş ve Kilo Arasındaki İlişki"""

def plot_yas_kilo(dataframe):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Yaş', y='Kilo', data=dataframe)
    plt.title('Yaş ve Kilo Arasındaki İlişki')
    plt.show()

plot_yas_kilo(df)

"""Kan Grubu ve Kronik Hastalıklar"""

def plot_kan_grubu_kronik(dataframe):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Kan Grubu', hue='Kronik Hastaliklarim', data=dataframe, palette='Set3')
    plt.title('Kan Grubuna Göre Kronik Hastalık Dağılımı')
    plt.xticks(rotation=45)
    plt.show()

plot_kan_grubu_kronik(df)

"""Yaş Grupları ve Yan Etkilerin Gözetilmesi"""

def plot_yas_gruplari_yan_etki(dataframe):
    bins = [0, 18, 30, 45, 60, 100]
    labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    dataframe['yas_gruplari'] = pd.cut(dataframe['Yaş'], bins=bins, labels=labels)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='yas_gruplari', hue='Yan_Etki', data=dataframe, palette='Set1')
    plt.title('Yaş Gruplarına Göre Yan Etki Dağılımı')
    plt.xticks(rotation=45)
    plt.show()

plot_yas_gruplari_yan_etki(df)

"""Cinsiyet ve İlaç Kullanılan Gün"""

def plot_cinsiyet_ilac_suresi_histogram(dataframe):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataframe, x='toplam_ilac_kullanilan_gun', hue='Cinsiyet', multiple='stack', palette='Set2')
    plt.title('Cinsiyete Göre İlaç Kullanım Süresi Dağılımı')
    plt.show()

plot_cinsiyet_ilac_suresi_histogram(df)

"""Kronik Hastalığı Olanlarda Yan Etki Dağılımı"""

def plot_kronik_yan_etki(dataframe):
    dataframe['Kronik_Hastalik_Varmi'] = dataframe['Kronik Hastaliklarim'].apply(lambda x: 'Evet' if x != 'Yok' else 'Hayır')

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Kronik_Hastalik_Varmi', hue='Yan_Etki', data=dataframe, palette='Set3')
    plt.title('Kronik Hastalığı Olanlarda Yan Etki Dağılımı')
    plt.show()

plot_kronik_yan_etki(df)

"""İllere Göre Yan Etki Dağılımı"""

def plot_il_yan_etki(dataframe):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Il', hue='Yan_Etki', data=dataframe, palette='Set3')
    plt.title('İllere Göre Yan Etki Dağılımı')
    plt.xticks(rotation=90)
    plt.show()

plot_il_yan_etki(df)

"""Korelasyon Haritası"""

numeric_df = df.select_dtypes(include=['float64', 'int64'])


def corr_map(df, width=14, height=6, annot_kws=15):
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(df.corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                vmin=-1,
                vmax=1,
                cmap="RdBu",
                mask=mtx,
                linewidth=0.4,
                linecolor="black",
                cbar=False,
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0, size=15)
    plt.xticks(rotation=75, size=15)
    plt.title('\nCorrelation Map\n', size=20)
    plt.show()


corr_map(numeric_df, width=20, height=10, annot_kws=8)

"""İlaç - Cinsiyet Kırılımında Ortalama Yaş, İlaç Kullanım Süresi ve Yan Etki Süresi:"""

def pivot_ilac_cinsiyet(df):
    pivot_table = df.pivot_table(index=['Ilac_Adi', 'Cinsiyet'],
                                 values=['Yaş', 'toplam_ilac_kullanilan_gun', 'Yan_etki_baslangic_tarihi'],
                                 aggfunc='mean')
    return pivot_table

pivot_ilac_cinsiyet(df)

"""Cinsiyet - Yaş Kırılımında Ortalama İlaç Kullanım Süresi ve Yan Etki Süresi:"""

def pivot_cinsiyet_yas(df):
    pivot_table = df.pivot_table(index=['Cinsiyet', 'Yaş'],
                                 values=['toplam_ilac_kullanilan_gun', 'Yan_etki_baslangic_tarihi'],
                                 aggfunc='mean')
    return pivot_table

pivot_cinsiyet_yas(df)

"""Kan Grubu - Yaş - Kilo Kırılımında İlaç Kullanım Süresi ve Yan Etki Süresi:"""

def kan_grubu_yas_kilo_analysis(df):
    result = df.groupby(['Kan Grubu', 'Yaş', 'Kilo']).apply(
        lambda x: pd.Series({
            'Ortalama_Ilac_Kullanim_Suresi': x['toplam_ilac_kullanilan_gun'].mean(),
            'Ortalama_Yan_Etki_Suresi': x['Yan_etki_baslangic_tarihi'].mean()
        })
    )
    return result


kan_grubu_yas_kilo_analysis(df)

"""Outlier Analizi"""

num_cols

df.info()

import pandas as pd

def check_outlier(dataframe, col_name):
    if pd.api.types.is_numeric_dtype(dataframe[col_name]):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return ((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)).sum() > 0
    else:
        return False

for col in df.select_dtypes(include=["float64", "int64"]).columns:
    print(col, check_outlier(df, col))

cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_col = [col for col in df[cat_cols].columns if df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_col:
    df = label_encoder(df, col)

"""Çoklu classa sahip kategorikler için one hot encoding uygulayalım"""

cat_cols = [col for col in cat_cols if col not in binary_col]
def one_hot_encoder(dataframe, categoric_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categoric_cols, drop_first=drop_first, dtype=int)
    return dataframe

df = one_hot_encoder(df,cat_cols,drop_first=True)

df.info()

df.head()