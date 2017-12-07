import numpy as np

def to_file(fileobj, uttids, embeddings, **kwargs):
    np.savez(fileobj, uttids=uttids, embeddings=embeddings, **kwargs)
