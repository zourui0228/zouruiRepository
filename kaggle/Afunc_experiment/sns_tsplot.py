import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
if 0:
    # build up the data
    df = []
    start_date = datetime.datetime(2015, 7, 1)
    for i in range(10):
        for j in [1,2]:
            unit = 'Ones' if j == 1 else 'Twos'
            date = start_date + datetime.timedelta(days=i)

            # I believe it makes more sense to directly convert the datetime to a
            # "matplotlib"-date (float), instead of creating strings and then let
            # pandas parse the string again
            df.append({
                    'Date': mdates.date2num(date),
                    'Value': i * j,
                    'Unit': unit
                })
    df = pd.DataFrame(df)
    print(df.head())
    print(df.info())

    # build the figure
    fig, ax = plt.subplots()
    sns.tsplot(data=df, time='Date', value='Value', unit='Unit', ax=ax)

    # assign locator and formatter for the xaxis ticks.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

    # put the labels at 45 deg since they tend to be too long
    fig.autofmt_xdate()
    plt.show()
if 1:
    gammas=sns.load_dataset('gammas')
    print(gammas.head())
    ax=sns.tsplot(data=gammas,time='timepoint',value='BOLD signal',unit='subject',condition='ROI')
    plt.show()
