
# Calculate % variance contribution of each component
# percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
#
# # Plot A Scree Plot that shows the percentage contribution of each PC component
# data_slice = percent_var[0:20]
#
# fig, (ax1, ax2) = plt.subplots(1, 2,  dpi=100, figsize=(20,8))
# fig.suptitle('PCA Explained Variances Plots')
#
# ax1.plot(np.cumsum(percent_var))
# ax1.set(xlabel='Cumulative Variance', ylabel='Principal Components')
# ax1.title.set_text(f'First 3 Components Explains {np.round(np.sum(data_slice[0:3]), decimals=1)}% of the data')
#
#
# ax2.bar(x=range(1, len(data_slice)+1), height=data_slice)
# ax2.set(xlabel="Percentage of Explained Variance", ylabel='Principal Components', xticks=range(1, len(data_slice)+1))
# ax2.title.set_text(f'{len(data_slice)} Components Explains {np.round(np.sum(data_slice), decimals=1)}% of the data')
# plt.show()
#
# pca_labels = [f'PC{str(x)}' for x in range(1, len(percent_var)+1)]
# pca_df = pd.DataFrame(pca_out, columns=pca_labels)
# pca_df.head()

#plot K-means, Jenk Breaks and KDE

# y = [i for i in range(len(manders_data))]
# plt.figure(2, figsize=(12,8), dpi= 100)
# ax = sns.distplot(manders_data_resh, hist=True, kde=True,norm_hist=True)
# patches = ax.patches
# for bar in patches:
#     color = None
#     if bar._x0 <= breaks[1]:
#         color = (102/255,194/255,165/255, .6)
#     elif  bar._x0 > breaks[1] and bar._x1 < breaks[2]:
#         color = (179/255,179/255,179/255, .6)
#     else:
#         color = (166/255,216/255,84/255, .6)
#
#     bar._facecolor = color
#
# plt.scatter(x=manders_data_resh, y=np.zeros_like(manders_data_resh) + 2,
#             c=cluster_labels, cmap='Set2')
# plt.vlines(breaks, 0, 2.1, colors='black', linestyle='dashed')
# plt.legend(['Kernal Density Estimator', '1D K-Means Clusters', '1D Jenkins Break Optimisation'])
# plt.title('1D Cluster Analysis of Mutant Cells')
# plt.xlabel('Manders Coefficient: Feature Standardised(mean=0, unit-variance)')
# for idx, cent in enumerate(centroids):
#     plt.annotate(f'X Centroid{idx}', (cent, 0))
#
# ax_x = plt.gca()
# leg = ax.get_legend()
# leg.legendHandles[1].set_color((166/255,216/255,84/255, .6))



# percentages = lambda x, totals: (x[0], np.round((x[1] / totals) * 100, decimals=1))
#
# mutants_per_cluster['cluster0']['counts'] = percentages(np.unique(np.asarray(mutants_per_cluster['cluster0']['data']), return_counts=True),
#                                                         mutant_counts)
# mutants_per_cluster['cluster1']['counts'] = percentages(np.unique(np.asarray(mutants_per_cluster['cluster1']['data']), return_counts=True),
#                                                         mutant_counts)
# mutants_per_cluster['cluster2']['counts'] = percentages(np.unique(np.asarray(mutants_per_cluster['cluster2']['data']), return_counts=True),
#                                                         mutant_counts)
#
# a = [str(x) for x in mutants_per_cluster['cluster0']['counts'][0]]