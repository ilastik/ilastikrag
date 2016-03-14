class BaseEdgeAccumulator(object):
    """
    Base class for all edge accumulators,
    i.e. accumulators that compute features from the edge values
    between superpixels (not superpixel contents).
    """

    #: Accumulator type
    ACCUMULATOR_TYPE = 'edge'

    #: An id string for this accumulator.
    #: Must not contain an underscore (``_``).
    #: Must not conflict with any other accumulators of the same type ('edge').
    #: All feature names supported by this accumulator must begin with the prefix ``<id>_edge_``
    ACCUMULATOR_ID = ''

    def __init__(self, rag, feature_names):
        """
        Parameters
        ----------
        rag:
            The rag.
        
        feature_names
            A list of feature names to compute with this accumulator.
        """
        pass
    
    def cleanup(self):
        """
        Called by the Rag to indicate that processing has completed, and
        the accumulator should discard all cached data and intermediate results.
        Subclasses must reimplement this function.
        """
        raise NotImplementedError

    @classmethod
    def supported_features(cls, rag):
        """
        Returns the list of feature names that can be computed for the given Rag.
        """
        raise NotImplementedError

    def ingest_edges(self, rag, edge_values):
        """
        Ingests the given edge values using the given Rag.

        Parameters
        ----------
        rag
            *Rag*
            
        edge_values
            *OrderedDict* of 1D *ndarray*.
            Each ndarray ``edge_values[k]`` is in the same order as ``rag.dense_edges[k]``
        """
        raise NotImplementedError
    
    def append_edge_features_to_df(self, edge_df):
        """
        Called by the Rag after ``ingest_edges()``.

        Merges the features of all ingested edges into a final set of edge
        feature columns, and appends those columns to the given
        ``pandas.DataFrame`` object.
        """        
        raise NotImplementedError
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()
