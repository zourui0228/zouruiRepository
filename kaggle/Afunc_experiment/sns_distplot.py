import numpy as np,seaborn as sns,matplotlib.pyplot as plt,pandas as pd
if 0:
    x=np.random.randn(10)
    sns.distplot(a=x,vertical=True)
    plt.show()
if 1:
    x1=np.random.randn(100)
    x1=pd.Series(data=x1,name='x1 variable')
    sns.distplot(a=x1,rug=True,hist=False)
    plt.show()
