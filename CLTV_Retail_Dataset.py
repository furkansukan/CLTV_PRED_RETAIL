##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# About Dataset
# Data holds the basic information about sales data.
# The company have sales agencies / resellers and branches and
# the data file holds only the branch/reseller information in the customer field.

# Veri Seti Hikayesi
# https://www.kaggle.com/datasets/shedai/retail-data-set


# Değişkenler
#
# InvoiceID : ID of the transaction. A transaction might hold multiple records for the same customer at the same date with multiple products (SKU). DocumentID might be useful for combining the transactions and detecting the items sold together.
# Date : Date of transaction / sell. In the date time format.
# ProductID : Item / Product code. The unique code for each item sold.
# TotalSales : Sales price for the transaction. If you want to get unit_price , divide TotalSales column to Quantity column
# Discount : Discount amount for the transaction.
# CustomerID : Unique customer id for each customer. For the data set, customer can be a reseller or a branch of the company.
# Quantity : Number of items sold in the transaction.

##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


#########################
# Görev 1: Veriyi Hazırlama
#########################

# ADIM 1 : Verinin Okunması


df_ = pd.read_csv("datasets/file_out2.csv")
df = df_.copy()

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(5))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Adım2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds
# ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

# Adım3: "TotalSales" değişkenini aykırı değerleri varsa baskılayanız.

replace_with_thresholds(df, "TotalSales")

df.describe().T

# Adım4: Unnamed değişkenini kaldır

df.drop("Unnamed: 0", axis=1, inplace=True)

# Adım5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

df["Date"] = df["Date"].astype("datetime64")

#########################
# Görev 2: CLTV Veri Yapısının Oluşturulması
#########################

# Adım1: Veri setindeki en son alışverişin yapıldığı tarihten
# 2 gün sonrasını analiz tarihi olarak alınız.

df["Date"].max() # Timestamp('2023-03-25 00:00:00')
today_date = dt.datetime(2023, 3, 27)

# Adım2: customer_id, recency_cltv_weekly, T_weekly,
# frequency ve monetary_cltv_avg değerlerinin yer
# aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak,
# recency ve tenure değerleri ise haftalık cinsten ifade edilecek.



cltv_df = df.groupby("CustomerID").agg({
    "Date" : [lambda Date: (Date.max() - Date.min()).days,
     lambda Date: (today_date - Date.min()).days],
    "InvoiceID": lambda InvoiceID: InvoiceID.nunique(),
    "TotalSales": lambda TotalSales: TotalSales.sum()
})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]

cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df = cltv_df[cltv_df["recency_cltv_weekly"] > 1]

cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7

cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7

cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"] / cltv_df["frequency"]

cltv_df[cltv_df["frequency"] > 1].shape
cltv_df[cltv_df["recency_cltv_weekly"] > 1].shape



#########################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#########################

# Adım1: BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

plot_period_transactions(bgf)
plt.show(block=True)

# Adım2: Gamma-Gamma modelini fit ediniz.
# Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# Adım3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6, # Aylık
                                   freq="W", # T'nin Frekans bilgisi
                                   discount_rate = 0.01)

# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv.sort_values(ascending=False).head(20)

#########################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
#########################

# Adım1: 6 aylık CLTV'ye göre tüm müşterilerinizi
# 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv"] = cltv

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.groupby("segment").agg({"mean", "median", "std"})


















