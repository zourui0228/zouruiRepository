import pandas as pd
import numpy as np
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
url='/home/zr/Documents/data/kaggle_data/k1_dgreen_house_price/'
df_train=pd.read_csv(url+'train.csv')
df_train_h=df_train.head(10)
# ---overall
if 0:
    df_t=df_train.head(10)
    #print(df_train.head())
    df_t_d=df_train.describe()
    print(df_train.describe())
    #print(df_train['SalePrice'].describe())
    sns.distplot(df_train['SalePrice'],kde=True,color='b',hist_kws={'alpha':0.9})
    plt.show()
#---numerical feature---
if 1:
    # pairwise correlation
    # t1=df_train.select_dtypes(exclude=['float','int']).head(10)

    corr=df_train.select_dtypes(include=['float','int']).iloc[:,1:].corr()
    if 1:
        plt.figure(figsize=(150,100))
        sns.heatmap(corr,vmax=1,square=True,annot=False
                    ,linewidths=2)
        plt.show()
        plt.close
    #t2=corr['SalePrice']
    t3=corr['SalePrice'].sort_values(ascending=False)
    del t3['SalePrice']
    print(t3)
    pass
    if 0:
        sns.regplot(x='OverallQual',y='SalePrice',data=df_train,x_estimator=np.mean)
        plt.show()

    # df.SalePrice=series
    # df.SalePrice.values=list
    price=df_train['SalePrice'].values
    # plt.figure(num=1,figsize=(10,9))
    if 0:
        f,ax_arr=plt.subplots(nrows=3,ncols=2,figsize=(10,9))
        ax_arr[0, 0].scatter(df_train.GrLivArea.values, price)
        ax_arr[0, 0].set_title('GrLiveArea')
        ax_arr[0, 1].scatter(df_train.GarageArea.values, price)
        ax_arr[0, 1].set_title('GarageArea')
        ax_arr[1, 0].scatter(df_train.TotalBsmtSF.values, price)
        ax_arr[1, 0].set_title('TotalBsmtSF')
        ax_arr[1, 1].scatter(df_train['1stFlrSF'].values, price)
        ax_arr[1, 1].set_title('1stFlrSF')
        ax_arr[2, 0].scatter(df_train.TotRmsAbvGrd.values, price)
        ax_arr[2, 0].set_title('TotRmsAbvGrd')
        ax_arr[2, 1].scatter(df_train.MasVnrArea.values, price)
        ax_arr[2, 1].set_title('MasVnrArea')
        f.text(0.00, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize=12)
        plt.tight_layout()
        plt.show()
    if 0:
        fig=plt.figure(num=2,figsize=(9,7))
        plt.subplot(211)
        plt.scatter(df_train.YearBuilt.values,price)
        plt.title(s='YearBuilt')

        plt.subplot(212)
        plt.scatter(df_train.YearRemodAdd.values, price)
        plt.title('YearRemodAdd')
        # va=vertical alignment
        # ha=horizontal alignment
        fig.text(x=0.01,y=0.5,s='SalePrice',verticalalignment='center',horizontalalignment='center',
                 rotation='vertical',fontsize=12)
        #plt.tight_layout()
        plt.show()
# ---categorical features
if 1:
    categorical_features=df_train.select_dtypes(include=['object'])
    categorical_features_h=categorical_features.head(n=10)
    # index(series)=df.columns
    #t4=categorical_features.columns
    # list  = series.values
    #t5=t4.values
    print(categorical_features.columns.values)
    if 0:
        plt.figure(figsize=(12,9))
        sns.boxplot(x='Neighborhood',y='SalePrice',data=df_train)
        plt.xticks(rotation=45)
        plt.show()
    if 0:
        plt.figure(figsize=(12,6))
        sns.countplot(x='Neighborhood',data=df_train)
        plt.xticks(rotation=45)
        plt.show()
    if 0:
        fig1,ax_arr1=plt.subplots(nrows=2,ncols=1,figsize=(10,6))
        sns.boxplot(x='SaleType',y='SalePrice',data=df_train,ax=ax_arr1[0])
        sns.boxplot(x='SaleCondition',y='SalePrice',data=df_train,ax=ax_arr1[1])
        plt.tight_layout()
        plt.show()
#
if 1:
    #month vs salePrice
    if 0:
        g=sns.FacetGrid(data=df_train,col='YrSold',col_wrap=3)
        g=g.map(sns.boxplot,'MoSold','SalePrice',order=range(1,13),palette='Set2')
        g.set(ylim=(0,500000))
        plt.show()
    #housing style && sale price
