from . import edges
from .node2vec import Node2Vec
from .graph2vec import Graph2Vec
from importlib import metadata

try:
    __version__ = metadata.version("graph2emb")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"


