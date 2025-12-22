"""
Tests for edge embedding functionality.
"""
import pytest
import networkx as nx
import numpy as np
from graph2emb import Node2Vec
from graph2emb.edges import (
    EdgeEmbedder,
    AverageEmbedder,
    HadamardEmbedder,
    WeightedL1Embedder,
    WeightedL2Embedder
)


class TestEdgeEmbedders:
    """Test cases for edge embedders."""
    
    @pytest.fixture(scope="module")
    def sample_model(self, small_graph, default_node2vec_params):
        """Create a sample Node2Vec model for testing (built once per module)."""
        node2vec = Node2Vec(small_graph, **default_node2vec_params)
        model = node2vec.fit(window=3, min_count=1, epochs=1)
        return model
    
    def test_average_embedder(self, sample_model):
        """Test AverageEmbedder."""
        embedder = AverageEmbedder(keyed_vectors=sample_model.wv, quiet=True)
        
        # Get a valid edge
        nodes = list(sample_model.wv.index_to_key)[:2]
        edge = (nodes[0], nodes[1])
        
        embedding = embedder[edge]
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == sample_model.wv.vector_size
        assert embedding.dtype in [np.float32, np.float64]
        
        # Average should be (node1 + node2) / 2
        expected = (sample_model.wv[nodes[0]] + sample_model.wv[nodes[1]]) / 2
        np.testing.assert_array_almost_equal(embedding, expected)
    
    def test_hadamard_embedder(self, sample_model):
        """Test HadamardEmbedder."""
        embedder = HadamardEmbedder(keyed_vectors=sample_model.wv, quiet=True)
        
        nodes = list(sample_model.wv.index_to_key)[:2]
        edge = (nodes[0], nodes[1])
        
        embedding = embedder[edge]
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == sample_model.wv.vector_size
        
        # Hadamard should be node1 * node2
        expected = sample_model.wv[nodes[0]] * sample_model.wv[nodes[1]]
        np.testing.assert_array_almost_equal(embedding, expected)
    
    def test_weighted_l1_embedder(self, sample_model):
        """Test WeightedL1Embedder."""
        embedder = WeightedL1Embedder(keyed_vectors=sample_model.wv, quiet=True)
        
        nodes = list(sample_model.wv.index_to_key)[:2]
        edge = (nodes[0], nodes[1])
        
        embedding = embedder[edge]
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == sample_model.wv.vector_size
        
        # L1 should be |node1 - node2|
        expected = np.abs(sample_model.wv[nodes[0]] - sample_model.wv[nodes[1]])
        np.testing.assert_array_almost_equal(embedding, expected)
        assert np.all(embedding >= 0)  # L1 norm should be non-negative
    
    def test_weighted_l2_embedder(self, sample_model):
        """Test WeightedL2Embedder."""
        embedder = WeightedL2Embedder(keyed_vectors=sample_model.wv, quiet=True)
        
        nodes = list(sample_model.wv.index_to_key)[:2]
        edge = (nodes[0], nodes[1])
        
        embedding = embedder[edge]
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == sample_model.wv.vector_size
        
        # L2 should be (node1 - node2)^2
        expected = (sample_model.wv[nodes[0]] - sample_model.wv[nodes[1]]) ** 2
        np.testing.assert_array_almost_equal(embedding, expected)
        assert np.all(embedding >= 0)  # L2 norm should be non-negative
    
    def test_edge_embedder_invalid_edge(self, sample_model):
        """Test that invalid edges raise appropriate errors."""
        embedder = AverageEmbedder(keyed_vectors=sample_model.wv, quiet=True)
        
        # Invalid edge format
        with pytest.raises(ValueError):
            embedder['invalid']
        
        with pytest.raises(ValueError):
            embedder[('single',)]
        
        # Non-existent nodes
        with pytest.raises(KeyError):
            embedder[('nonexistent1', 'nonexistent2')]
    
    def test_as_keyed_vectors(self, sample_model):
        """Test as_keyed_vectors method."""
        embedder = HadamardEmbedder(keyed_vectors=sample_model.wv, quiet=True)
        
        edge_kv = embedder.as_keyed_vectors()
        
        assert edge_kv is not None
        assert edge_kv.vector_size == sample_model.wv.vector_size
        
        # Check that edges are stored as sorted tuples
        nodes = list(sample_model.wv.index_to_key)
        if len(nodes) >= 2:
            edge_key = str(tuple(sorted((nodes[0], nodes[1]))))
            assert edge_key in edge_kv.index_to_key
    
    def test_all_embedder_types(self, sample_model):
        """Test that all embedder types work correctly."""
        nodes = list(sample_model.wv.index_to_key)[:2]
        edge = (nodes[0], nodes[1])
        
        embedders = [
            AverageEmbedder(sample_model.wv, quiet=True),
            HadamardEmbedder(sample_model.wv, quiet=True),
            WeightedL1Embedder(sample_model.wv, quiet=True),
            WeightedL2Embedder(sample_model.wv, quiet=True),
        ]
        
        for embedder in embedders:
            embedding = embedder[edge]
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == sample_model.wv.vector_size
    
    def test_edge_embedder_with_different_node_types(self, sample_model):
        """Test edge embedder with string node names."""
        nodes = list(sample_model.wv.index_to_key)[:2]
        
        # Test with string nodes
        edge1 = (nodes[0], nodes[1])
        edge2 = (nodes[1], nodes[0])  # Reverse order
        
        embedder = AverageEmbedder(keyed_vectors=sample_model.wv, quiet=True)
        
        # Both should work
        emb1 = embedder[edge1]
        emb2 = embedder[edge2]
        
        # Average should be the same regardless of order
        np.testing.assert_array_almost_equal(emb1, emb2)

