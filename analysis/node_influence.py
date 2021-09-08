import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os


def calc_top_percent(betweenness_list, percent):
    height = int(len(betweenness_list) * percent)
    sum_top_betweenness = 0
    sum_total_betweenness = sum(betweenness_list)
    for i in range(height):
        sum_top_betweenness += betweenness_list[i]

    top_percent_betweenness = sum_top_betweenness / sum_total_betweenness
    print(top_percent_betweenness)
    return top_percent_betweenness


def plot_line(betweenness_percentages, percentage, dates):
    plt.figure()
    # timestamps = [str(i) for i in timestamps]
    df_plot = pd.DataFrame({'bt_percentage': betweenness_percentages}, index=dates)

    ax = df_plot.plot.area(ylim=(0.65, 1), rot=20, alpha=0.65, stacked=False, color=['red'])
    # Annotate
    for i in range(len(dates)):
        ax.annotate('{:.2f}'.format(betweenness_percentages[i] * 100) + '%', xytext=(i - 0.3, betweenness_percentages[i] + 0.05),
                    xy=(i, betweenness_percentages[i]), arrowprops=dict(arrowstyle='->'), fontsize=8, color='black')

    plt.xticks()
    plt.yticks()
    ax.set_xlabel('Timestamps')
    ax.set_ylabel('Percentage')
    ax.get_legend().remove()
    plt.title('Top ' + str(percentage*100) + '% nodes betweenness share' )
    Path("plots/top").mkdir(parents=True, exist_ok=True)
    filePath = cwd + '/plots/top/top_line_' + str(percentage) + '.png'
    plt.savefig(filePath, bbox_inches='tight', dpi=400)
    plt.show()


timestamps = [
    1554112800,
    1564653600,
    1572606000,
    1585735200,
    1596276000,
    1606820400,
    1609498800
]

# timestamp = timestamps[3]

top_betweenness_percentages = list()
percentage = 0.10
dates = ['01.Apr 2019', '01.Aug 2019', '01.Nov 2019', '01.Apr 2020', '01.Aug 2020', '01.Dec 2020', '01.Jan 2021']
for timestamp in timestamps:
    baseAmount = [10000000, 1000000000, 10000000000]

    cwd = str(Path().resolve())
    filepath = cwd + '/' + str(timestamp) + '/' + str(baseAmount[0])
    filenames = next(os.walk(filepath), (None, None, []))[2]  # [] if no file
    df = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[0]) + '/' + filenames[3])
    df_2 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[1]) + '/' + filenames[3])
    df_3 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[2]) + '/' + filenames[3])

    betweenness = (list(filter(lambda a: a != 0.0, df['betweenness'])))
    betweenness_2 = (list(filter(lambda a: a != 0.0, df_2['betweenness'])))
    betweenness_3 = (list(filter(lambda a: a != 0.0, df_3['betweenness'])))

    avg_betwenness = list()
    for b1, b2, b3 in zip(betweenness, betweenness_2, betweenness_3):
        avg_betwenness.append(np.average([b1, b2, b3]))

    top_betweenness_percentages.append(calc_top_percent(avg_betwenness, percentage))
plot_line(top_betweenness_percentages, percentage, dates)
