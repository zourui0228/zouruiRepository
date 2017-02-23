import seaborn as sns
tips=sns.load_dataset('tips')
tips_h=tips.head(10)
print(tips.columns.values)
sns.regplot(x='total_bill',y='tip',data=tips,fit_reg=False)
sns.plt.show()
x=0