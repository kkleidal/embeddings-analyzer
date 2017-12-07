import argparse
import numpy as np
from .encoder import to_file

def producer_gmm(**kwargs):
    k = kwargs.get("k", 3)
    D = kwargs["dims"]
    centers = np.random.normal(loc=0.0, scale=10.0, size=[k, D])
    pis = np.random.dirichlet(10 * np.ones(k))
    accuracy = 0.8
    n = kwargs["n"]
    assignments = np.random.choice(k, size=n, p=pis)
    flips = np.random.choice(2, size=n, p=[accuracy, 1 - accuracy])
    reassignments = np.random.choice(k, size=n, p=pis).astype(np.bool)
    guesses = np.where(flips, reassignments, assignments)
    mus = np.take(centers, assignments, axis=0)
    x = np.random.normal(loc=mus, scale=1.0, size=[n, D])
    to_file(kwargs["outfile"], np.array(list(range(n))), x, ground_truth=assignments, predictions=guesses)

producers = {name[len("producer_"):]: val for name, val in globals().items() if name.startswith("producer_")}

def make_toy_data(**kwargs):
    producers[kwargs["producer"]](**kwargs)

def add_arguments(parser):
    # parser.add_argument('--pca', metavar='dimensions', type=int, default=None, 'Dimensionality reduction with PCA')
    parser.add_argument('--dims', metavar='dimensions', type=int, default=3, help='Dimensionality of data')
    parser.add_argument('--producer', metavar='producer', type=str, default="gmm", choices=producers.keys(),
            help='Function to produce data')
    parser.add_argument('n', type=int, help='Number of examples')
    parser.add_argument('outfile', type=argparse.FileType('wb'), help='Output file')

