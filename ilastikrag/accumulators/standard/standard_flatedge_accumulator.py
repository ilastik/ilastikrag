import logging
import numpy as np
import pandas as pd
import vigra

from ilastikrag.accumulators import BaseFlatEdgeAccumulator
from .vigra_util import get_vigra_feature_names, append_vigra_features_to_dataframe

logger = logging.getLogger(__name__)

class StandardFlatEdgeAccumulator(BaseFlatEdgeAccumulator):
    """
    Accumulator for basic features of 'flat edges',
    i.e. the edges in the Z-direction when ``Rag.flat_superpixels == True``.
    
    Supported feature names:

        - standard_flatedge_count
        - standard_flatedge_sum
        - standard_flatedge_minimum
        - standard_flatedge_maximum
        - standard_flatedge_mean
        - standard_flatedge_variance
        - standard_flatedge_kurtosis
        - standard_flatedge_skewness

    ..

        - standard_flatedge_quantiles (short for "all sp quantiles")
        - standard_flatedge_quantiles_0
        - standard_flatedge_quantiles_10
        - standard_flatedge_quantiles_25
        - standard_flatedge_quantiles_50
        - standard_flatedge_quantiles_75
        - standard_flatedge_quantiles_90
        - standard_flatedge_quantiles_100

    ..

        - standard_flatedge_regionradii (both of the below)
        - standard_flatedge_regionradii_0
        - standard_flatedge_regionradii_1

    ..

        - standard_flatedge_regionaxes (all of the below)
        - standard_flatedge_regionaxes_0x
        - standard_flatedge_regionaxes_0y
        - standard_flatedge_regionaxes_1x
        - standard_flatedge_regionaxes_1y

    ..
        
        - standard_flatedge_correlation
    """
    ACCUMULATOR_ID = 'standard'
    ACCUMULATOR_TYPE = 'flatedge'
    
    def __init__(self, rag, feature_names):
        self.cleanup() # Initialize members
        feature_names = list(feature_names)

        # 'standard_flatedge_quantiles' is shorthand for "all quantiles"
        if 'standard_flatedge_quantiles' in feature_names:
            feature_names.remove('standard_flatedge_quantiles')
            
            feature_names += ['standard_flatedge_quantiles_0',
                              'standard_flatedge_quantiles_10',
                              'standard_flatedge_quantiles_25',
                              'standard_flatedge_quantiles_50',
                              'standard_flatedge_quantiles_75',
                              'standard_flatedge_quantiles_90',
                              'standard_flatedge_quantiles_100']

        # 'standard_flatedge_regionradii' is shorthand for "all regionradii"
        if 'standard_flatedge_regionradii' in feature_names:
            feature_names.remove('standard_flatedge_regionradii')
            feature_names += ['standard_flatedge_regionradii_0',
                              'standard_flatedge_regionradii_1']
        
        # 'standard_flatedge_regionaxes' is shorthand for "all regionaxes"
        if 'standard_flatedge_regionaxes' in feature_names:
            feature_names.remove('standard_flatedge_regionaxes')
            feature_names += ['standard_flatedge_regionaxes_0x',
                              'standard_flatedge_regionaxes_0y',
                              'standard_flatedge_regionaxes_1x',
                              'standard_flatedge_regionaxes_1y']
        
        self._feature_names = feature_names
        self._vigra_feature_names = get_vigra_feature_names(feature_names)
    
    def cleanup(self):
        self._vigra_acc = None

    def ingest_values(self, rag, value_img):
        if value_img is None:
            assert self._vigra_feature_names == ['count'], \
                "Can't compute flatedge features without a value image (except for standard_flatedge_count)"

        if value_img is not None:
            # Convert to float32 if necessary
            value_img = value_img.astype(np.float32, copy=False)
            value_img = (value_img[1:] + value_img[:-1]) / 2.
        else:
            for feat in self._vigra_feature_names:
                assert feat.startswith('region') or feat == 'count', \
                    "Can't compute feature {} without a value image!"
            
            # Vigra wants a value image, even though we won't be using it.
            # We'll give it some garbage:
            # Just cast the labels as if they were float.
            value_img = rag.label_img[:-1].view(np.float32)
            value_img = vigra.taggedView(value_img, rag.label_img.axistags)

        self._vigra_acc = vigra.analysis.extractRegionFeatures( value_img,
                                                                rag.flat_edge_label_img,
                                                                features=self._vigra_feature_names,
                                                                histogramRange="globalminmax" )

    def append_edge_features_to_df(self, edge_df):
        # Add the vigra accumulator results to the dataframe
        return append_vigra_features_to_dataframe(self._vigra_acc, edge_df, self._feature_names, overwrite_quantile_minmax=True)

    @classmethod
    def supported_features(cls, rag):
        names = ['standard_flatedge_count',
                 'standard_flatedge_sum',
                 'standard_flatedge_minimum',
                 'standard_flatedge_maximum',
                 'standard_flatedge_mean',
                 'standard_flatedge_variance',
                 'standard_flatedge_kurtosis',
                 'standard_flatedge_skewness',
                 'standard_flatedge_quantiles',
                 'standard_flatedge_quantiles_0',
                 'standard_flatedge_quantiles_10',
                 'standard_flatedge_quantiles_25',
                 'standard_flatedge_quantiles_50',
                 'standard_flatedge_quantiles_75',
                 'standard_flatedge_quantiles_90',
                 'standard_flatedge_quantiles_100',
                 'standard_flatedge_regionradii',
                 'standard_flatedge_regionradii_0',
                 'standard_flatedge_regionradii_1',
                 'standard_flatedge_regionaxes',
                 'standard_flatedge_regionaxes_0x',
                 'standard_flatedge_regionaxes_0y',
                 'standard_flatedge_regionaxes_1x',
                 'standard_flatedge_regionaxes_1y']
        return names
