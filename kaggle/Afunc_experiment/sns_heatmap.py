import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
if 0:
    uniform_data=np.random.rand(3,5)
    ax=sns.heatmap(data=uniform_data,vmin=0,vmax=2)
    plt.show()
if 0:
    flights=sns.load_dataset(name='flights')
    flights_h=flights.head()
    flights=flights.pivot(index='month',columns='year',values='passengers')
    sns.heatmap(data=flights,annot=True,fmt='d',linewidths=0.5)
    plt.show()
if 1:
    data=np.random.rand(20,50)
    ax=sns.heatmap(data=data,xticklabels=5,yticklabels=2)
    plt.show()