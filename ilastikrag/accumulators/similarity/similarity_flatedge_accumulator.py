import logging
import numpy as np
import pandas as pd

from ilastikrag.util import edge_ids_for_axis
from ilastikrag.accumulators import BaseFlatEdgeAccumulator

logger = logging.getLogger(__name__)

class SimilarityFlatEdgeAccumulator(BaseFlatEdgeAccumulator):
    """
    Accumulator for computing various similarity metrics between pixel values adjacent to 'flat edges',
    i.e. the edges in the Z-direction when ``Rag.flat_superpixels == True``.
    
    Supported feature names:
        
        - similarity_flatedge_correlation
    """
    ACCUMULATOR_ID = 'similarity'
    ACCUMULATOR_TYPE = 'flatedge'
    
    def __init__(self, rag, feature_names):
        self.cleanup() # Initialize members
        feature_names = list(feature_names)
        self._feature_names = feature_names
    
    def cleanup(self):
        self._final_df = None

    def ingest_values(self, rag, value_img):
        self._final_df = pd.DataFrame(rag.unique_edge_tables['z'][['sp1', 'sp2']])
         
        if 'similarity_flatedge_correlation' in self._feature_names:
            self._compute_correlation_feature(rag, value_img)
            pass
        
    def append_edge_features_to_df(self, edge_df):
        merged = pd.merge(edge_df, self._final_df, how='left', on=['sp1', 'sp2'])
        return merged

    def _compute_correlation_feature(self, rag, value_img):
        """
        Compute the correlation between edge-adjacent pixels and append it to final_df
        """
        # FIXME: We should be able to avoid this sort until the end...
        z_edge_ids = edge_ids_for_axis(rag.label_img, None, 0)
        z_edge_ids.sort(1)
        
        values_df = pd.DataFrame(z_edge_ids, columns=['sp1', 'sp2'])
        values_df['left_values'] = np.array(value_img[:-1].reshape(-1))
        values_df['right_values'] = np.array(value_img[1:].reshape(-1))
        
        correlations = np.zeros( len(self._final_df), dtype=np.float32 )
        
        group_index = [0]
        def write_correlation(group_df):
            """
            Computes the correlation between 'left_values' and 'right_values' of the given group,
            and writes it into the pre-existing correlations_array.
            """
            # Compute
            covariance = np.cov(group_df['left_values'].values, group_df['right_values'].values)
            denominator = np.sqrt(covariance[0,0]*covariance[1,1])
            if denominator == 0.0:
                corr = 1.0
            else:
                assert covariance[0,1] == covariance[1,0]
                corr = (covariance[0,1] / denominator)
            
            # Store
            correlations[group_index[0]] = corr
            group_index[0] += 1
            
            # We don't need to return anything;
            # we're using this function only for its side effects.
            return None

        grouper = values_df.groupby(['sp1', 'sp2'], sort=True, group_keys=False)
        grouper.apply(write_correlation) # Used for its side-effects only
        
        self._final_df['similarity_flatedge_correlation'] = pd.Series(correlations, dtype=np.float32, index=self._final_df.index)

    @classmethod
    def supported_features(cls, rag):
        if not rag.flat_superpixels:
            return []
        names = ['similarity_flatedge_correlation']
        return names
