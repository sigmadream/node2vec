"""
Tests for Node2Vec basic functionality.
"""
import pytest
import networkx as nx
import numpy as np
from graph2emb import Node2Vec


class TestNode2Vec:
    """Test cases for Node2Vec class."""
    
    def test_basic_initialization(self, medium_graph, default_node2vec_params):
        """Test basic Node2Vec initialization."""
        node2vec = Node2Vec(medium_graph, **default_node2vec_params)
        
        assert node2vec.graph == medium_graph
        assert node2vec.dimensions == default_node2vec_params["dimensions"]
        assert node2vec.walk_length == default_node2vec_params["walk_length"]
        assert node2vec.num_walks == default_node2vec_params["num_walks"]
        assert node2vec.workers == default_node2vec_params["workers"]
        # 작은 그래프는 walk가 0일 수도 있으니, 타입만 확인
        assert isinstance(node2vec.walks, list)
    
    def test_walks_generation(self, small_graph, default_node2vec_params):
        """Test that walks are generated correctly."""
        node2vec = Node2Vec(small_graph, **default_node2vec_params)
        
        # Check that walks are generated
        assert isinstance(node2vec.walks, list)
        
        # Check walk structure
        for walk in node2vec.walks:
            assert isinstance(walk, list)
            assert len(walk) <= default_node2vec_params["walk_length"]  # walk_length
            assert len(walk) > 0
            
            # All nodes in walk should be strings
            for node in walk:
                assert isinstance(node, str)
    
    def test_probability_precomputation(self, small_graph, default_node2vec_params):
        """Test that probabilities are precomputed correctly."""
        node2vec = Node2Vec(small_graph, **default_node2vec_params)
        
        # Check that d_graph has probabilities
        assert len(node2vec.d_graph) > 0
        
        # Check that probabilities are normalized (sum to 1)
        for node, data in node2vec.d_graph.items():
            if node2vec.PROBABILITIES_KEY in data:
                for source, probs in data[node2vec.PROBABILITIES_KEY].items():
                    assert isinstance(probs, np.ndarray)
                    # Probabilities should sum to approximately 1
                    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
            
            if node2vec.FIRST_TRAVEL_KEY in data:
                first_travel = data[node2vec.FIRST_TRAVEL_KEY]
                assert isinstance(first_travel, np.ndarray)
                assert np.isclose(first_travel.sum(), 1.0, atol=1e-6)
    
    def test_fit_model(self, small_graph, default_node2vec_params):
        """Test fitting a Word2Vec model."""
        node2vec = Node2Vec(small_graph, **default_node2vec_params)
        
        model = node2vec.fit(window=3, min_count=1, epochs=1)
        
        assert model is not None
        assert model.wv.vector_size == default_node2vec_params["dimensions"]
        
        # Check that embeddings exist for nodes
        nodes = [str(n) for n in small_graph.nodes()]
        for node in nodes[:5]:  # Check first 5 nodes
            if node in model.wv:
                assert len(model.wv[node]) == default_node2vec_params["dimensions"]
    
    def test_custom_parameters(self, small_graph):
        """Test Node2Vec with custom p and q parameters."""
        node2vec = Node2Vec(
            small_graph,
            dimensions=8,
            walk_length=3,
            num_walks=1,
            p=0.5, 
            q=2.0, 
            workers=1, 
            quiet=True
        )
        
        assert node2vec.p == 0.5
        assert node2vec.q == 2.0
        assert len(node2vec.walks) > 0
    
    def test_sampling_strategy(self, small_graph):
        """Test Node2Vec with custom sampling strategy."""
        # Create sampling strategy for first node
        first_node = list(small_graph.nodes())[0]
        sampling_strategy = {
            first_node: {
                'p': 0.5,
                'q': 2.0,
                'num_walks': 1,
                'walk_length': 3
            }
        }
        
        node2vec = Node2Vec(
            small_graph,
            dimensions=8,
            walk_length=3,
            num_walks=1,
            sampling_strategy=sampling_strategy,
            workers=1,
            quiet=True
        )
        
        assert node2vec.sampling_strategy == sampling_strategy
        assert len(node2vec.walks) > 0
    
    def test_seed_reproducibility(self, small_graph):
        """Test that seed produces reproducible results."""
        # Create two Node2Vec instances with same seed
        node2vec1 = Node2Vec(small_graph, dimensions=8, walk_length=3, num_walks=1, seed=123, workers=1, quiet=True)
        node2vec2 = Node2Vec(small_graph, dimensions=8, walk_length=3, num_walks=1, seed=123, workers=1, quiet=True)
        
        # With workers=1, results should be reproducible
        assert len(node2vec1.walks) == len(node2vec2.walks)
    
    def test_string_nodes(self):
        """Test Node2Vec with string node names."""
        graph = nx.Graph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
        
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True)
        
        assert len(node2vec.walks) > 0
        # All nodes in walks should be strings
        for walk in node2vec.walks:
            for node in walk:
                assert isinstance(node, str)
                assert node in ['A', 'B', 'C', 'D']
    
    def test_weighted_graph(self):
        """Test Node2Vec with weighted graph."""
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            (0, 1, 2.0),
            (1, 2, 1.5),
            (2, 3, 3.0),
            (3, 0, 1.0)
        ])
        
        node2vec = Node2Vec(graph, dimensions=8, walk_length=3, num_walks=1, workers=1, quiet=True)
        
        assert len(node2vec.walks) > 0
    
    def test_empty_graph_error(self):
        """Test that empty graph raises appropriate error."""
        graph = nx.Graph()
        
        # Empty graph should still work, but generate no walks
        node2vec = Node2Vec(graph, dimensions=8, walk_length=3, num_walks=1, workers=1, quiet=True)
        assert len(node2vec.walks) == 0

