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

            feature_names += ['standard_edge_quantiles_0',
                              'standard_edge_quantiles_10',
                              'standard_edge_quantiles_25',
                              'standard_edge_quantiles_50',
                              'standard_edge_quantiles_75',
                              'standard_edge_quantiles_90',
                              'standard_edge_quantiles_100']

        self._feature_names = feature_names
        self._vigra_feature_names = get_vigra_feature_names(feature_names)

    def cleanup(self):
        self._vigra_acc = None
    
    def ingest_edges(self, rag, edge_values):
        """
        Create a vigra RegionFeaturesAccumulator with the results of all features,
        computed over the edge pixels of the given value_img.

        The accumulator's 'region' indexes will correspond to the 'edge_label'
        column from the given DataFrames.
        """
        if edge_values is None:
            assert self._vigra_feature_names == ['count'], \
                "Can't compute edge features without a value image (except for standard_edge_count)"

        # Compute histogram_range across all axes (if quantiles are needed)
        if set(['quantiles', 'histogram']) & set(self._vigra_feature_names):
            logger.debug("Computing global histogram range...")
            histogram_range = [min(map(np.min, edge_values.values())),
                               max(map(np.max, edge_values.values()))]
            if histogram_range:
                logger.warning("All edge pixels are identical. Is this image empty?")
                histogram_range[1] += 1.0
        else:
            histogram_range = "globalminmax"

        axial_accumulators = []
        for axiskey, dense_edge_table in rag.dense_edge_tables.items():
            logger.debug("Axis {}: Computing region features...".format( axiskey ))
            
            edge_labels = dense_edge_table['edge_label'].values
            if edge_values:            
                edge_values_thisaxis = edge_values[axiskey]
            else:
                # Vigra wants a value image, even though we won't be using it.
                # We'll give it some garbage:
                # Just cast the labels as if they were float.
                edge_values_thisaxis = edge_labels.view(np.float32)
        
            # Must add an extra singleton axis here because vigra doesn't support 1D data
            acc = vigra.analysis.extractRegionFeatures( edge_values_thisaxis.reshape((1,-1), order='A'),
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
        self._vigra_acc = final_acc

    def append_edge_features_to_df(self, edge_df):
        # Add the vigra accumulator results to the dataframe
        return append_vigra_features_to_dataframe(self._vigra_acc, edge_df, self._feature_names, overwrite_quantile_minmax=True)
    
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
    
