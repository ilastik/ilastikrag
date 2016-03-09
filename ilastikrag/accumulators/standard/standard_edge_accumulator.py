import logging
import numpy as np
import vigra

from ilastikrag.accumulators import BaseEdgeAccumulator
from .vigra_util import get_vigra_feature_names, append_vigra_features_to_dataframe

logger = logging.getLogger(__name__)

class StandardEdgeAccumulator(BaseEdgeAccumulator):
    """
    Accumulator for features of the edge pixels (only) between superpixels.
    Uses vigra's RegionFeatureAccumulator library to compute the features.
    
    Supported feature names:
    
        - standard_edge_count
        - standard_edge_sum
        - standard_edge_minimum
        - standard_edge_maximum
        - standard_edge_mean
        - standard_edge_variance
        - standard_edge_kurtosis
        - standard_edge_skewness

    ..

        - standard_edge_quantiles (short for "all edge quantiles")
        - standard_edge_quantiles_0
        - standard_edge_quantiles_10
        - standard_edge_quantiles_25
        - standard_edge_quantiles_50
        - standard_edge_quantiles_75
        - standard_edge_quantiles_90
        - standard_edge_quantiles_100
    """

#     TODO
#     ----    
#     - edge_count is computed 'manhattan' style, meaning that it
#       is sensitive to the edge orientation (and so is edge_sum).
#       Should we try to compensate for that somehow?

    
    ACCUMULATOR_ID = 'standard'
    ACCUMULATOR_TYPE = 'edge'

    def __init__(self, rag, feature_names):
        self.cleanup() # Initialize members
        feature_names = list(feature_names)

        # 'standard_edge_quantiles' is shorthand for "all quantiles"
        if 'standard_edge_quantiles' in feature_names:
            feature_names.remove('standard_edge_quantiles')

            # Quantile histogram_range is based on the first block only,
            # so the '0' and '100' values would be misleading if we used them as-is.
            # Instead, we silently replace the '0' and '100' quantiles with 'minimum' and 'maximum'.
            # See vigra_util.append_vigra_features_to_dataframe()
            feature_names += ['standard_edge_quantiles_0',  # Will be automatically replaced with 'minimum'
                              'standard_edge_quantiles_10',
                              'standard_edge_quantiles_25',
                              'standard_edge_quantiles_50',
                              'standard_edge_quantiles_75',
                              'standard_edge_quantiles_90',
                              'standard_edge_quantiles_100'] # Will be automatically replaced with 'maximum'

        self._feature_names = feature_names
        self._vigra_feature_names = get_vigra_feature_names(feature_names)

    def cleanup(self):
        self._histogram_range = None
        self._block_vigra_accumulators = []
    
    def ingest_edges_for_block(self, axial_edge_dfs, block_start, block_stop):
        assert len(self._block_vigra_accumulators) == 0, \
            "FIXME: This accumulator is written to support block-wise accumulation, "\
            "but that use-case isn't tested yet.\n"\
            "Write a unit test for that use-case, then remove this assertion."
        
        block_vigra_acc = self._accumulate_edge_vigra_features( axial_edge_dfs )
        self._block_vigra_accumulators.append( block_vigra_acc )

    def append_merged_edge_features_to_df(self, edge_df):
        # Merge all the accumulators from each block
        final_acc = self._block_vigra_accumulators[0].createAccumulator()
        for block_vigra_acc in self._block_vigra_accumulators:
            # This is an identity lookup, but it's necessary since vigra will complain 
            # about different maxIds if we call merge() without a lookup 
            axis_to_final_index_array = np.arange( block_vigra_acc.maxRegionLabel()+1, dtype=np.uint32 )
            final_acc.merge( block_vigra_acc, axis_to_final_index_array )
        
        # Add the vigra accumulator results to the dataframe
        edge_df = append_vigra_features_to_dataframe(final_acc, edge_df, self._feature_names, overwrite_quantile_minmax=True)
        return edge_df
    
    def _accumulate_edge_vigra_features(self, axial_edge_dfs):
        """
        Return a vigra RegionFeaturesAccumulator with the results of all features,
        computed over the edge pixels of the given value_img.

        The accumulator's 'region' indexes will correspond to the 'edge_label'
        column from the given DataFrames.
        
        If this is the first block of data we've seen, initialize the histogram range.
        """
        # Compute histogram_range across all axes of the first block (if quantiles are needed)
        if self._histogram_range is None and set(['quantiles', 'histogram']) & set(self._vigra_feature_names):
            logger.debug("Computing global histogram range...")
            histogram_range = [min(map(lambda df: df['edge_value'].min(), axial_edge_dfs)),
                               max(map(lambda df: df['edge_value'].max(), axial_edge_dfs))]
            
            # Cache histogram_range for subsequent blocks
            # Technically, this means that the range of the first block
            # is used for all other blocks, too, but that's necessary for merging
            # the results in get_final_features_df()
            self._histogram_range = histogram_range

        if self._histogram_range is None:
            histogram_range = "globalminmax"

        axial_accumulators = []
        for axis, axial_edge_df in enumerate( axial_edge_dfs ):
            logger.debug("Axis {}: Computing region features...".format( axis ))

            edge_labels = axial_edge_df['edge_label'].values
            edge_values = axial_edge_df['edge_value'].values
        
            # Must add an extra singleton axis here because vigra doesn't support 1D data
            acc = vigra.analysis.extractRegionFeatures( edge_values.reshape((1,-1), order='A'),
                                                        edge_labels.reshape((1,-1), order='A'),
                                                        features=self._vigra_feature_names,
                                                        histogramRange=histogram_range )
            axial_accumulators.append(acc)

        final_acc = axial_accumulators[0].createAccumulator()
        for acc in axial_accumulators:
            # This is an identity lookup, but it's necessary since vigra will complain 
            # about different maxIds if we call merge() without a lookup 
            axis_to_final_index_array = np.arange( acc.maxRegionLabel()+1, dtype=np.uint32 )
            final_acc.merge( acc, axis_to_final_index_array )
        return final_acc

    @classmethod
    def supported_features(cls, rag):
        names = ['standard_edge_count',
                 'standard_edge_sum',
                 'standard_edge_minimum',
                 'standard_edge_maximum',
                 'standard_edge_mean',
                 'standard_edge_variance',
                 'standard_edge_kurtosis',
                 'standard_edge_skewness',
                 'standard_edge_quantiles',
                 'standard_edge_quantiles_0',
                 'standard_edge_quantiles_10',
                 'standard_edge_quantiles_25',
                 'standard_edge_quantiles_50',
                 'standard_edge_quantiles_75',
                 'standard_edge_quantiles_90',
                 'standard_edge_quantiles_100' ]
        return names
    
