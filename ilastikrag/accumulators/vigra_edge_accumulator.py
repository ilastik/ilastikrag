import logging
import numpy as np
import vigra

from ..edge_accumulator import EdgeAccumulator
from .vigra_util import get_vigra_feature_names, append_vigra_features_to_dataframe

logger = logging.getLogger(__name__)

class VigraEdgeAccumulator(EdgeAccumulator):
    """
    Accumulator for features of the edge pixels (only) between superpixels.
    Uses vigra's RegionFeatureAccumulator library to compute the features.
    
    Supported feature names:
    
        - edge_vigra_count
        - edge_vigra_sum
        - edge_vigra_minimum
        - edge_vigra_maximum
        - edge_vigra_mean
        - edge_vigra_variance
        - edge_vigra_kurtosis
        - edge_vigra_skewness
        - edge_vigra_quantiles_0
        - edge_vigra_quantiles_10
        - edge_vigra_quantiles_25
        - edge_vigra_quantiles_50
        - edge_vigra_quantiles_75
        - edge_vigra_quantiles_90
        - edge_vigra_quantiles_100
        
    Coordinate-based features (such as RegionAxes) are not supported yet.
    """
    ACCUMULATOR_TYPE = 'edge'
    ACCUMULATOR_ID = 'vigra'
    
    def __init__(self, label_img, feature_names):
        self.cleanup() # Initialize members
        self._feature_names = feature_names
        self._vigra_feature_names = get_vigra_feature_names(feature_names)

    def cleanup(self):
        self._histogram_range = None
        self._block_vigra_accumulators = []
    
    def ingest_edges_for_block(self, axial_edge_dfs, block_start, block_stop):
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
        edge_df = append_vigra_features_to_dataframe(final_acc, edge_df, self._feature_names)
        return edge_df
    
    def _accumulate_edge_vigra_features(self, axial_edge_dfs):
        """
        Return a vigra RegionFeaturesAccumulator with the results of all features,
        computed over the edge pixels of the given value_img.

        The accumulator's 'region' indexes will correspond to the 'edge_label'
        column from the given DataFrames.
        
        If this is the first block of data we've seen, initialize the histogram range.
        """
        for feature_name in self._vigra_feature_names:
            for nonsupported_name in ('coord', 'region'):
                # We can't use vigra to compute coordinate-based features because 
                # we've already flattened the edge pixels into a 1D array.
                # However, the coordinates are already recorded in the axial_edge_df,
                # so it would be easy to compute RegionRadii directly, without vigra.
                assert nonsupported_name not in feature_name.lower(), \
                    "Coordinate-based edge features are not currently supported!"

        # If we need to compute quantiles,
        # we first need to find the histogram_range to use
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
            edge_labels = axial_edge_df['edge_label'].values
            edge_values = axial_edge_df['edge_value'].values
        
            logger.debug("Axis {}: Computing region features...".format( axis ))
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
