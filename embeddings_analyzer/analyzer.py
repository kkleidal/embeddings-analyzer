import sys
import argparse
import numpy as np
import sklearn.decomposition
import sklearn.manifold
import matplotlib
from IPython.display import display
import sklearn.metrics
import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from collections import Counter
import sklearn.cluster

markers= [
    "o",
	"s",
	"x",
	"*",
	"D",
	"v",
	"^",
	"<",
	">",
	"1",
	"2",
	"3",
	"4",
	"8",
	"p",
	"P",
	"h",
	"H",
	"+",
	"d",
	"X",
	"|",
	"_"]

def analysis_projection(uttids, embeddings, **kwargs):
    plt = kwargs["plt"]
    pca = kwargs.get("pca", None)
    if pca is not None and pca > 0:
        pca_model = sklearn.decomposition.pca(n_components=pca)
        embeddings = pca_model.fit_transform(embeddings)
    tsne = kwargs.get("tsne", True)
    if tsne:
        tsne_model = sklearn.manifold.TSNE(n_components=2)
        embeddings = tsne_model.fit_transform(embeddings)
    if embeddings.shape[1] != 2:
        raise RuntimeError("Insufficient projections. Not 2D")
    extras = kwargs.get("extras", {})
    if kwargs.get("keep_percentage", 1.0) < 1.0:
        N = embeddings.shape[0]
        p = kwargs.get("keep_percentage", 1.0)
        np.random.seed(99)
        kept = np.random.choice(range(N), size=int(p * N), replace=False)
        uttids = uttids[kept]
        embeddings = embeddings[kept]
        extras = {name: vals[kept] for name, vals in extras.items()}
    c = None
    cmap = None 
    if kwargs.get("color_by", None) is not None:
        c = extras[kwargs["color_by"]]
        cmap = "Paired" if c.dtype.kind in 'ui' else "Spectral"
        norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
        c = matplotlib.cm.get_cmap(cmap)(norm(c))
    s = None
    if kwargs.get("size_by", None) is not None:
        s = extras[kwargs["size_by"]]
        s = s - s.min()
        s = 9 * s / s.max()
        s = s + 2
        s = s * s
        s = np.round(s).astype(np.int32)
    if kwargs.get("shape_by", None) is not None:
        shape_data = extras[kwargs["shape_by"]]
        shape_vals = sorted(set(shape_data))
        assert len(shape_vals) < len(markers)
        shape_map = {val: markers[i] for i, val in enumerate(shape_vals)}
        marker_map = {}
        for i, shape in enumerate(shape_data):
            marker = shape_map[shape]
            if marker not in marker_map:
                marker_map[marker] = []
            marker_map[marker].append(i)
    else:
        marker_map = {"o": list(range(embeddings.shape[0]))}
    fig = plt.figure()
    for marker, idces in marker_map.items():
        pltkwargs = {}
        if s is not None:
            pltkwargs["s"] = s[idces]
        if c is not None:
            pltkwargs["c"] = c[idces]
        plt.scatter(embeddings[idces,0], embeddings[idces,1], marker=marker, **pltkwargs, edgecolor='black')
    if kwargs.get("interactive", False):
        # Interactive
        plt.show()
    else:
        # Non-interactive
        fig.savefig(kwargs["output"], format='png')
    plt.close(fig)

def analysis_extras(uttids, embeddings, **kwargs):
    for extra in sorted(kwargs.get("extras", {}).keys()):
        print(extra)

class _SimilarityMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.extras = kwargs.get("extras", {})
        self.items = [(name, self.extras[name]) for name in sorted(self.extras.keys()) if self.extras[name].dtype.kind in 'ui']
        self.M = len(self.items)

    def display_similarity_table(self, table):
        kwargs = self.kwargs
        items = self.items
        M = len(items)
        if kwargs.get("interactive", False):
            DF_var = pd.DataFrame.from_items((items[i][0], table[i,:]) for i in range(M))
            DF_var.index = [key for key, _ in items]
            display(DF_var)
        else:
            kwargs["output"].write(("name\t" + "\t".join(item[0] for item in items) + "\n").encode('utf8'))
            for i in range(M):
                kwargs["output"].write((("%s\t" % items[i][0]) + "\t".join("%f" % table[i, j] for j in range(M)) + "\n").encode('utf8'))

    def display_purity_table(self, table):
        kwargs = self.kwargs
        items = self.items
        M = len(items)
        if kwargs.get("interactive", False):
            DF_var = pd.DataFrame.from_items((items[i][0], [table[i]]) for i in range(M))
            DF_var.index = ["Cluster purity"]
            display(DF_var)
        else:
            kwargs["output"].write(("name\t" + "\t".join(item[0] for item in items) + "\n").encode('utf8'))
            for i in range(M):
                kwargs["output"].write((("%s\t" % items[i][0]) + "\t".join("%f" % table[i, j] for j in range(M)) + "\n").encode('utf8'))

def analysis_nmi(uttids, embeddings, **kwargs):
    metric = _SimilarityMetric(**kwargs)
    M = metric.M
    items = metric.items
    nmi = np.zeros([M, M])
    for i in range(M):
        for j in range(i, M):
            nmi[i, j] = sklearn.metrics.normalized_mutual_info_score(items[i][1], items[j][1])
            nmi[j, i] = nmi[i, j]
    metric.display_similarity_table(nmi)
    
def analysis_cluster_purity(uttids, embeddings, **kwargs):
    metric = _SimilarityMetric(**kwargs)
    M = metric.M
    items = metric.items
    purity_table = np.zeros([M])
    if kwargs.get("clustering", None) is not None:
        clusters = next(vals for item, vals in items if item == kwargs["clustering"])
    else:
        k = kwargs["k"]
        clustering = sklearn.cluster.KMeans(k)
        clustering.fit(embeddings)
        clusters = clustering.predict(embeddings)
    for j in range(M):
        labels = items[j][1]
        N = clusters.shape[0]
        purity = 0
        for cluster in set(clusters):
            indices = np.array([i for i, c in enumerate(clusters) if c == cluster])
            correct = max(Counter(labels[indices]).values())
            purity += correct
        purity /= N
        purity_table[j] = purity
    metric.display_purity_table(purity_table)
    if kwargs.get("verbose", False):
        display("Cluster centers:")
        display(clustering.cluster_centers_)
        display("Clustering:")
        display(clusters)

analysis = {name[len("analysis_"):]: val for name, val in globals().items() if name.startswith("analysis_")}

def analyze(**kwargs):
    inputs = dict(np.load(kwargs["input"]))
    try:
        uttids = inputs["uttids"]
        embeddings = inputs["embeddings"]
        del inputs["uttids"]
        del inputs["embeddings"]
    except KeyError:
        raise RuntimeError("Bad embeddings file.")
    del kwargs["input"]
    kwargs["extras"] = inputs
    analysis[kwargs["analysis"]](uttids, embeddings, **kwargs)

def add_arguments(parser):
    # parser.add_argument('--pca', metavar='dimensions', type=int, default=None, 'Dimensionality reduction with PCA')
    parser.add_argument('--dims', metavar='dimensions', type=int, default=3, help='Dimensionality of data')
    parser.add_argument('--analysis', type=str, default="projection", choices=analysis.keys(),
            help='Analysis to run')
    parser.add_argument('-i', '--interactive', action='store_true', 
            help='If you want results displayed interactively rather than saved')
    parser.add_argument('--pca', type=int, default=None, help='Perform PCA before TSNE for projections')
    parser.add_argument('--no-tsne', dest='tsne', action='store_false', help='Do not perform TSNE for projection')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Show additional information')
    parser.add_argument('--color-by', type=str, default=None, help='Color by this extra for projection')
    parser.add_argument('--shape-by', type=str, default=None, help='Shape by this extra for projection')
    parser.add_argument('--size-by', type=str, default=None, help='Size by this extra for projection')
    parser.add_argument('--keep-percentage', type=float, default=1.0, help='Keep this percentage of points while plotting projection')
    parser.add_argument('-k', '--k', type=int, default=None, help='k for kmeans clustering')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'), default=sys.stdout.buffer, help='Output file')
    parser.add_argument('input', metavar='input-file', type=argparse.FileType('rb'), help='Input embeddings file')

