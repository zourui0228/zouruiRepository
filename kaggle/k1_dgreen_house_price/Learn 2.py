# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#njobs = 4
# Get data
url='/home/zr/Documents/data/kaggle_data/k1_dgreen_house_price/'
train=pd.read_csv(url+'train.csv')
train_h=train.head(10)
if 0:
    # GrLivArea>4000 drop
    fig,ax_arr=plt.subplots(nrows=2,ncols=1,sharex=True)
    sns.regplot(x='GrLivArea',y='SalePrice',data=train,fit_reg=True,ax=ax_arr[0])
    train=train[train.GrLivArea<4000]
    # GrLivArea>4000 drop
    sns.regplot(x='GrLivArea',y='SalePrice',data=train,fit_reg=True,ax=ax_arr[1])
    plt.show()
train=train[train.GrLivArea<4000]
if 0:
    # log1p(SalePrice)
    fig,ax_arr1=plt.subplots(nrows=2,ncols=1,sharex=False)
    sns.regplot(x='GrLivArea',y='SalePrice',data=train,fit_reg=True,ax=ax_arr1[0])
    train.SalePrice=np.log1p(train.SalePrice)
    sns.regplot(x='GrLivArea',y='SalePrice',data=train,fit_reg=True,ax=ax_arr1[1])
    plt.yticks()
    plt.ylim(np.min(train.SalePrice),np.max(train.SalePrice))
    plt.show()
train.SalePrice=np.log1p(train.SalePrice)
if 1:
    print('---before---')
    corr=train.corr()
    corr.sort_values(by=['SalePrice'],axis=0,ascending=False,inplace=True)
    print('before'+str(corr['SalePrice']))
    #----train_num, train_category----
    train_num=train.select_dtypes(include=['float','int']).iloc[:,1:]
    train_cat=train.select_dtypes(include=['object'])
    print('before_num feature:'+str(len(train_num.columns.values)))
    print('before_cat feature:'+str(len(train_cat.columns.values)))
    if 1:
        # Handle missing values for features where median/mean or most common value doesn't make sense

        # Alley : data description says NA means "no alley access"
        train["Alley"] = train["Alley"].fillna("None")
        # BedroomAbvGr : NA most likely means 0
        train["BedroomAbvGr"] = train["BedroomAbvGr"].fillna(0)
        # BsmtQual etc : data description says NA for basement features is "no basement"
        train["BsmtQual"] = train["BsmtQual"].fillna("No")
        train["BsmtCond"] = train["BsmtCond"].fillna("No")
        train["BsmtExposure"] = train["BsmtExposure"].fillna("No")
        train["BsmtFinType1"] = train["BsmtFinType1"].fillna("No")
        train["BsmtFinType2"] = train["BsmtFinType2"].fillna("No")
        train["BsmtFullBath"] = train["BsmtFullBath"].fillna(0)
        train["BsmtHalfBath"] = train["BsmtHalfBath"].fillna(0)
        train["BsmtUnfSF"] = train["BsmtUnfSF"].fillna(0)
        # CentralAir : NA most likely means No
        train["CentralAir"] = train["CentralAir"].fillna("N")
        # Condition : NA most likely means Normal
        train["Condition1"] = train["Condition1"].fillna("Norm")
        train["Condition2"] = train["Condition2"].fillna("Norm")
        # EnclosedPorch : NA most likely means no enclosed porch
        train["EnclosedPorch"] = train["EnclosedPorch"].fillna(0)
        # External stuff : NA most likely means average
        train["ExterCond"] = train["ExterCond"].fillna("TA")
        train["ExterQual"] = train["ExterQual"].fillna("TA")
        # Fence : data description says NA means "no fence"
        train["Fence"] = train["Fence"].fillna("No")
        # FireplaceQu : data description says NA means "no fireplace"
        train["FireplaceQu"] = train["FireplaceQu"].fillna("No")
        train["Fireplaces"] = train["Fireplaces"].fillna(0)
        # Functional : data description says NA means typical
        train["Functional"] = train["Functional"].fillna("Typ")
        # GarageType etc : data description says NA for garage features is "no garage"
        train["GarageType"] = train["GarageType"].fillna("No")
        train["GarageFinish"] = train["GarageFinish"].fillna("No")
        train["GarageQual"] = train["GarageQual"].fillna("No")
        train["GarageCond"] = train["GarageCond"].fillna("No")
        train["GarageArea"] = train["GarageArea"].fillna(0)
        train["GarageCars"] = train["GarageCars"].fillna(0)
        # HalfBath : NA most likely means no half baths above grade
        train["HalfBath"] = train["HalfBath"].fillna(0)
        # HeatingQC : NA most likely means typical
        train["HeatingQC"] = train["HeatingQC"].fillna("TA")
        # KitchenAbvGr : NA most likely means 0
        train["KitchenAbvGr"] = train["KitchenAbvGr"].fillna(0)
        # KitchenQual : NA most likely means typical
        train["KitchenQual"] = train["KitchenQual"].fillna("TA")
        # LotFrontage : NA most likely means no lot frontage
        train["LotFrontage"] = train["LotFrontage"].fillna(0)
        # LotShape : NA most likely means regular
        train["LotShape"] = train["LotShape"].fillna("Reg")
        # MasVnrType : NA most likely means no veneer
        train["MasVnrType"] = train["MasVnrType"].fillna("None")
        train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
        # MiscFeature : data description says NA means "no misc feature"
        train["MiscFeature"] = train["MiscFeature"].fillna("No")
        train["MiscVal"] = train["MiscVal"].fillna(0)
        # OpenPorchSF : NA most likely means no open porch
        train["OpenPorchSF"] = train["OpenPorchSF"].fillna(0)
        # PavedDrive : NA most likely means not paved
        train["PavedDrive"] = train["PavedDrive"].fillna("N")
        # PoolQC : data description says NA means "no pool"
        train["PoolQC"] = train["PoolQC"].fillna("No")
        train["PoolArea"] = train["PoolArea"].fillna(0)
        # SaleCondition : NA most likely means normal sale
        train["SaleCondition"] = train["SaleCondition"].fillna("Normal")
        # ScreenPorch : NA most likely means no screen porch
        train["ScreenPorch"] = train["ScreenPorch"].fillna(0)
        # TotRmsAbvGrd : NA most likely means 0
        train["TotRmsAbvGrd"] = train["TotRmsAbvGrd"].fillna(0)
        # Utilities : NA most likely means all public utilities
        train["Utilities"] = train["Utilities"].fillna("AllPub")
        # WoodDeckSF : NA most likely means no wood deck
        train["WoodDeckSF"] = train["WoodDeckSF"].fillna(0)
        #----some numerical feature --> categorical feature----
        train.replace({'MoSold':{1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'},
                       'MSSubClass': {20: 'SC20', 30: 'SC30', 40: 'SC40', 45: 'SC45',
                                      50: 'SC50', 60: 'SC60', 70: 'SC70', 75: 'SC75',
                                      80: 'SC80', 85: 'SC85', 90: 'SC90', 120: 'SC120',
                                      150: 'SC150', 160: 'SC160', 180: 'SC180', 190: 'SC190'}})
        #----some categorical feature --> numerial feature----
        train = train.replace({'Alley': {'Grvl': 1, 'Pave': 2},
                               'BsmtCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3},
                               'BsmtFinType1': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,
                                                'ALQ': 5, 'GLQ': 6},
                               'BsmtFinType2': {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,
                                                'ALQ': 5, 'GLQ': 6},
                               'BsmtQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'FireplaceQu': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5,
                                              'Min2': 6, 'Min1': 7, 'Typ': 8},
                               'GarageCond': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'GarageQual': {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                               'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},
                               'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
                               'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
                               'PoolQC': {'No': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                               'Street': {'Grvl': 1, 'Pave': 2},
                               'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}}
                              )
        #----create new featrue-----
        # 1* Simplifications of existing features
        train['SimplOverallQual'] = train.OverallQual.replace({1: 1, 2: 1, 3: 1,  # bad
                                                               4: 2, 5: 2, 6: 2,  # average
                                                               7: 3, 8: 3, 9: 3, 10: 3  # good
                                                               })
        train['SimplOverallCond'] = train.OverallCond.replace({1: 1, 2: 1, 3: 1,  # bad
                                                               4: 2, 5: 2, 6: 2,  # average
                                                               7: 3, 8: 3, 9: 3, 10: 3  # good
                                                               })
        train['SimplPoolQC'] = train.PoolQC.replace({1: 1, 2: 1,  # average
                                                     3: 2, 4: 2  # good
                                                     })
        train['SimplGarageCond'] = train.GarageCond.replace({1: 1,  # bad
                                                             2: 1, 3: 1,  # average
                                                             4: 2, 5: 2  # good
                                                             })
        train['SimplGarageQual'] = train.GarageQual.replace({1: 1,  # bad
                                                             2: 1, 3: 1,  # average
                                                             4: 2, 5: 2  # good
                                                             })
        train['SimplFireplaceQu'] = train.FireplaceQu.replace({1: 1,  # bad
                                                               2: 1, 3: 1,  # average
                                                               4: 2, 5: 2  # good
                                                               })
        train['SimplFireplaceQu'] = train.FireplaceQu.replace({1: 1,  # bad
                                                               2: 1, 3: 1,  # average
                                                               4: 2, 5: 2  # good
                                                               })
        train['SimplFunctional'] = train.Functional.replace({1: 1, 2: 1,  # bad
                                                             3: 2, 4: 2,  # major
                                                             5: 3, 6: 3, 7: 3,  # minor
                                                             8: 4  # typical
                                                             })
        train['SimplKitchenQual'] = train.KitchenQual.replace({1: 1,  # bad
                                                               2: 1, 3: 1,  # average
                                                               4: 2, 5: 2  # good
                                                               })
        train['SimplHeatingQC'] = train.HeatingQC.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
        train['SimplBsmtFinType1'] = train.BsmtFinType1.replace({1: 1,  # unfinished
                                                                 2: 1, 3: 1,  # rec room
                                                                 4: 2, 5: 2, 6: 2  # living quarters
                                                                 })
        train['SimplBsmtFinType2'] = train.BsmtFinType2.replace({1: 1,  # unfinished
                                                                 2: 1, 3: 1,  # rec room
                                                                 4: 2, 5: 2, 6: 2  # living quarters
                                                                 })
        train['SimplBsmtCond'] = train.BsmtCond.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
        train['SimplBsmtQual'] = train.BsmtQual.replace({1: 1,  # bad
                                                         2: 1, 3: 1,  # average
                                                         4: 2, 5: 2  # good
                                                         })
        train['SimplExterCond'] = train.ExterCond.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })
        train['SimplExterQual'] = train.ExterQual.replace({1: 1,  # bad
                                                           2: 1, 3: 1,  # average
                                                           4: 2, 5: 2  # good
                                                           })

        # 2* Combinations of existing features
        # Overall quality of the house
        train['OverallGrade'] = train['OverallQual'] * train['OverallCond']
        # Overall quality of the garage
        train['GarageGrade'] = train['GarageQual'] * train['GarageCond']
        # Overall quality of the exterior
        train['ExterGrade'] = train['ExterQual'] * train['ExterCond']
        # Overall kitchen score
        train['KitchenScore'] = train['KitchenAbvGr'] * train['KitchenQual']
        # Overall fireplace score
        train['FireplaceScore'] = train['Fireplaces'] * train['FireplaceQu']
        # Overall garage score
        train['GarageScore'] = train['GarageArea'] * train['GarageQual']
        # Overall pool score
        train['PoolScore'] = train['PoolArea'] * train['PoolQC']
        # Simplified overall quality of the house
        train['SimplOverallGrade'] = train['SimplOverallQual'] * train['SimplOverallCond']
        # Simplified overall quality of the exterior
        train['SimplExterGrade'] = train['SimplExterQual'] * train['SimplExterCond']
        # Simplified overall pool score
        train['SimplPoolScore'] = train['PoolArea'] * train['SimplPoolQC']
        # Simplified overall garage score
        train['SimplGarageScore'] = train['GarageArea'] * train['SimplGarageQual']
        # Simplified overall fireplace score
        train['SimplFireplaceScore'] = train['Fireplaces'] * train['SimplFireplaceQu']
        # Simplified overall kitchen score
        train['SimplKitchenScore'] = train['KitchenAbvGr'] * train['SimplKitchenQual']
        # Total number of bathrooms
        train['TotalBath'] = train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']) + \
                             train['FullBath'] + (0.5 * train['HalfBath'])
        # Total SF for house (incl. basement)
        train['AllSF'] = train['GrLivArea'] + train['TotalBsmtSF']
        # Total SF for 1st + 2nd floors
        train['AllFlrsSF'] = train['1stFlrSF'] + train['2ndFlrSF']
        # Total SF for porch
        train['AllPorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + \
                              train['3SsnPorch'] + train['ScreenPorch']
        # Has masonry veneer or not
        train['HasMasVnr'] = train.MasVnrType.replace({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1,
                                                       'Stone': 1, 'None': 0})
        # House completed before sale or not
        train['BoughtOffPlan'] = train.SaleCondition.replace({'Abnorml': 0, 'Alloca': 0, 'AdjLand': 0,
                                                              'Family': 0, 'Normal': 0, 'Partial': 1})
        #-----create new feature-----
        # 3* Polynomials on the top 10 existing features
        train['OverallQual-s2'] = train['OverallQual'] ** 2
        train['OverallQual-s3'] = train['OverallQual'] ** 3
        train['OverallQual-Sq'] = np.sqrt(train['OverallQual'])
        train['AllSF-2'] = train['AllSF'] ** 2
        train['AllSF-3'] = train['AllSF'] ** 3
        train['AllSF-Sq'] = np.sqrt(train['AllSF'])
        train['AllFlrsSF-2'] = train['AllFlrsSF'] ** 2
        train['AllFlrsSF-3'] = train['AllFlrsSF'] ** 3
        train['AllFlrsSF-Sq'] = np.sqrt(train['AllFlrsSF'])
        train['GrLivArea-2'] = train['GrLivArea'] ** 2
        train['GrLivArea-3'] = train['GrLivArea'] ** 3
        train['GrLivArea-Sq'] = np.sqrt(train['GrLivArea'])
        train['SimplOverallQual-s2'] = train['SimplOverallQual'] ** 2
        train['SimplOverallQual-s3'] = train['SimplOverallQual'] ** 3
        train['SimplOverallQual-Sq'] = np.sqrt(train['SimplOverallQual'])
        train['ExterQual-2'] = train['ExterQual'] ** 2
        train['ExterQual-3'] = train['ExterQual'] ** 3
        train['ExterQual-Sq'] = np.sqrt(train['ExterQual'])
        train['GarageCars-2'] = train['GarageCars'] ** 2
        train['GarageCars-3'] = train['GarageCars'] ** 3
        train['GarageCars-Sq'] = np.sqrt(train['GarageCars'])
        train['TotalBath-2'] = train['TotalBath'] ** 2
        train['TotalBath-3'] = train['TotalBath'] ** 3
        train['TotalBath-Sq'] = np.sqrt(train['TotalBath'])
        train['KitchenQual-2'] = train['KitchenQual'] ** 2
        train['KitchenQual-3'] = train['KitchenQual'] ** 3
        train['KitchenQual-Sq'] = np.sqrt(train['KitchenQual'])
        train['GarageScore-2'] = train['GarageScore'] ** 2
        train['GarageScore-3'] = train['GarageScore'] ** 3
        train['GarageScore-Sq'] = np.sqrt(train['GarageScore'])
    #-----after modify feature, what is the corr and feature num/cat distribution-----
    print('---after---')
    corr=train.corr()
    corr.sort_values(by=['SalePrice'],axis=0,ascending=False,inplace=True)
    print(corr['SalePrice'])
    #----train_num, train_category----
    train_num=train.select_dtypes(include=['float','int']).drop(labels='SalePrice',axis=1)
    train_cat=train.select_dtypes(include=['object'])
    print('after_num feature:'+str(len(train_num.columns.values)))
    print('after_cat feature:'+str(len(train_cat.columns.values)))
    #fill na
    print(train_num.isnull().values.sum())
    train_num=train_num.fillna(train_num.median())
    print(train_num.isnull().values.sum())
if 1:
    # what is skew?
    skew