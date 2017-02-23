import seaborn as sns
import matplotlib.pyplot as plt
tips=sns.load_dataset('tips')
print(tips.head())
if 1:
    g=sns.FacetGrid(data=tips,row='smoker',col='time')
    g.map(plt.hist,'total_bill')
    plt.show()
if 1:
    g=sns.FacetGrid(data=tips,row='smoker',col='time')
    g.map(plt.scatter,'total_bill','tip',edgecolor='w')
    plt.show()
if 1:
    g=sns.FacetGrid(data=tips,col='time',hue='smoker')
    g=g.map(plt.scatter,'total_bill','tip',edgecolor='w')
    g.add_legend()
    plt.show()
if 1:
    g=sns.FacetGrid(data=tips,col='day',size=4,aspect=0.5)
    g.map(sns.boxplot,'time','total_bill')
    plt.show()
if 1:
    g=sns.FacetGrid(data=tips,col='smoker',col_order=['No','Yes'])
    g.map(plt.hist,'total_bill',color='m')
    plt.show()