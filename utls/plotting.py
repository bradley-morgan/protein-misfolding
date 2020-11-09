import matplotlib.pyplot as plt
import seaborn as sns


def density(data):

    plt.figure(0)
    plt.hist(data[:, 20], bins=40)

    plt.figure(1)
    sns.distplot(data[:, 20], hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    plt.show()