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
    #print(df_train.describe())
    #print(df_train['SalePrice'].describe())
    sns.distplot(df_train['SalePrice'],kde=True,color='b',hist_kws={'alpha':0.9})
    plt.show()
#---numerical feature---
if 1:
    # pairwise correlation
    #t1=df_train.select_dtypes(exclude=['float','int']).head(10)
    corr=df_train.select_dtypes(include=['float','int']).iloc[:,1:].corr()
    if 0:
        plt.figure(figsize=(150,100))
        sns.heatmap(data=corr,vmax=1,square=True,annot=False,fmt='.1g',linewidths=2)
        plt.show()
    #t2=corr['SalePrice']
    t3=corr['SalePrice'].sort_values(ascending=False)
    del t3['SalePrice']
    print(t3)
    if 1:
        sns.regplot(x='OverallQual',y='SalePrice',data=df_train,x_estimator=np.mean)
        plt.show()
    if 1:
        pass
    # df.SalePrice=series
    # df.SalePrice.values=list
    price=df_train['SalePrice'].values
    # plt.figure(num=1,figsize=(10,9))
    if 1:
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
