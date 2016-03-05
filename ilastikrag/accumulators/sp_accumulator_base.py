class SpAccumulatorBase(object):
    """
    Base class for all superpixel accumulators,
    i.e. accumulators that compute features from the
    contents of superpixels (not their edges).
    """
    
    #: Accumulator type
    ACCUMULATOR_TYPE = 'sp'

    #: An id string for this accumulator.
    #: Must not contain an underscore (``_``).
    #: Must not conflict with any other accumulators of the same type ('sp').
    #: All feature names supported by this accumulator must begin with the prefix ``<id>_sp_``
    ACCUMULATOR_ID = ''

    def __init__(self, label_img, feature_names):
        """
        Parameters
        ----------
        label_img
            The Rag's full label volume.
        
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

    def ingest_values_for_block(self, label_block, value_block, block_start, block_stop):
        """
        Ingests a particular block of label data and its corresponding (single-channel) pixel values.
        
        Parameters
        ----------
        label_block
            *VigraArray*, ``uint32``
        
        value_block
            *VigraArray*
        
        block_start
            The location of the block within the Rag's full label volume.
        
        block_stop
            The end of the block within the Rag's full label volume.
        """
        raise NotImplementedError
    
    def append_merged_sp_features_to_edge_df(self, edge_df):
        """
        Called by the Rag after all blocks have been ingested.

        Merges the features of all ingested blocks into a final set of edge
        feature columns, and appends those columns to the given
        ``pandas.DataFrame`` object.
        
        This involves converting pairs superpixel features into edge features,
        typically by taking the sum and/or difference between the features of
        each superpixel in an adjacent pair.
        """        
        raise NotImplementedError

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()
