import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap
import pandas as pd
import tools.general_tools as g_tools
import wandb

def make_k_means_labels(K, data: np.ndarray):
    km = KMeans(n_clusters=K)
    cluster_labels = km.fit_predict(data)
    centroids = km.cluster_centers_

    return cluster_labels, centroids


def optim_kmeans(data, K=21, max_iter=300):
    # Silhoutte score
    sil_scores = []
    with tqdm(total=K, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}') as progress_bar:
        for k in range(2, K):
            km = KMeans(n_clusters=k, max_iter=max_iter)
            cluster_labels = km.fit_predict(data)
            sil = silhouette_score(data, cluster_labels)
            sil_scores.append(sil)
            progress_bar.update(1)

    sns.set()
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(12, 8))
    fig.suptitle('K-Means Finding Optimal K Clusters')
    ax1.plot(range(2, K), sil_scores, '.-')
    ax1.vlines(np.argmax(sil_scores) + 2, min(sil_scores), max(sil_scores), colors='grey', linestyle='dashed')
    ax1.title.set_text('Silhouette Method')
    ax1.set(xlabel='Number of Clusters', ylabel='Silhouette Score', xticks=range(2, K))

    # elbow score
    sum_of_sqrd_distances = []
    with tqdm(total=K, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}') as progress_bar:
        for k in range(1, K):
            km = KMeans(n_clusters=k, max_iter=max_iter)
            km = km.fit(data)
            sum_of_sqrd_distances.append(km.inertia_)
            progress_bar.update(1)

    ax2.plot(range(1, K), sum_of_sqrd_distances, '.-')
    ax2.title.set_text('Elbow Method for Optimal K')
    ax2.set(xlabel='Number of Clusters', ylabel='Sum of Squared Distances', xticks=range(1, K))
    plt.show()


def visualise_cluster_distribution(data, centroids, k_means_labels, x_axis_name):
    # Create Plots to overlay K-means, Jenks and Kernal Density
    sns.set()
    y = [i for i in range(len(data))]
    plt.figure(2, figsize=(12, 8), dpi=100)
    ax = sns.distplot(data, hist=True, kde=True, norm_hist=True, bins=30)
    plt.scatter(x=data, y=np.zeros_like(data) + 2,
                c=k_means_labels, cmap='Set2')
    plt.legend(['Kernal Density Estimator', '1D K-Means Clusters'])
    plt.title('1D Cluster Analysis')
    plt.xlabel(x_axis_name)
    for idx, cent in enumerate(centroids):
        plt.annotate(f'X Centroid{idx}', (cent, 0))

    ax_x = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[1].set_color((166 / 255, 216 / 255, 84 / 255, .6))
    plt.show()


def correlate_shaps(shap, X, feature_names):
    # NEEDS CORRECTING FOR MIS FOLDING DATA
    # import matplotlib as plt
    # Make a copy of the input data
    shap = pd.DataFrame(shap, columns=feature_names)
    X = pd.DataFrame(shap, columns=feature_names)
    feature_order_df = pd.DataFrame(shap, columns=feature_names).abs().mean(axis=0).sort_values(ascending=False)
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_names:
        corr = np.corrcoef(shap[i], X[i])[1][0]

        corr_list.append(corr)

    corr_df = pd.concat([pd.Series(feature_names), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Feature', 'Corr']
    corr_df.insert(2, 'Sign', np.where(corr_df['Corr'] > 0, 'Pos', 'Neg'))

    corr_df.set_index('Feature', inplace=True)
    correl = corr_df.loc[feature_order_df.index.values]
    correl.insert(0, 'Mean Absolute SHAP Importance', feature_order_df)
    correl.reset_index(inplace=True)
    correl.index.name = 'Order of Importance'

    return correl


def shap_feature_importance(model, x_train, y_train, feature_name,  run=None):

    shap_vales = shap.TreeExplainer(model).shap_values(x_train, y_train)
    neg_correls = correlate_shaps(shap_vales[0], x_train, feature_names=feature_name)
    pos_correls = correlate_shaps(shap_vales[1], x_train, feature_names=feature_name)


    if not run:
        return shap_vales

    run.log({'Neg Correlations': wandb.Table(dataframe=neg_correls)})
    run.log({'Pos Correlations': wandb.Table(dataframe=pos_correls)})

    image_saver = g_tools.ImageSaver(run)

    sns.set()
    plt.figure(0)
    shap.summary_plot(
        shap_values=shap_vales[0],
        features=x_train,
        feature_names=feature_name,
        show=False,
        plot_type='dot',
        title='Neg Class'
    )
    plt.tight_layout()
    image_saver.save(plt.gcf(), 'Local Feature Importance: Negative Class', 'png')

    plt.figure(1)
    shap.summary_plot(
        shap_values=shap_vales[1],
        features=x_train,
        feature_names=feature_name,
        show=False,
        plot_type='dot',
        title='Pos Class'
    )
    plt.tight_layout()
    image_saver.save(plt.gcf(), 'Local Feature Importance: Positive Class', 'png')


    plt.figure(2)
    shap.summary_plot(
        shap_vales,
        x_train,
        plot_type='bar',
        feature_names=feature_name,
        show=False
    )
    plt.show()
    image_saver.save(plt.gcf(), 'Global Feature Importance', 'png')



    # fig2 = shap.dependence_plot(
    #     '193/CB - /1/C Hydrophobic',
    #     shap_vales,
    #     x_train,
    #     feature_names=feature_name,
    #     x_jitter=0.05,
    #     color='#000000',
    #     show=False
    # )

    return shap_vales