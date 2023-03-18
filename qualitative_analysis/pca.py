# TODO: Possibly plot PCA between context and no-context embeddings?
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(df)
X = pca.transform(df)
plot = plt.scatter(X[:,0], X[:,1], c=y)
#    plt.legend(handles=plot.legend_elements()[0], labels=list(winedata['target_names']))
plt.show()