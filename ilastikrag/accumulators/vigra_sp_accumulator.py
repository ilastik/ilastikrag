import logging
import numpy as np
import pandas as pd
import vigra

from ..sp_accumulator import SpAccumulator
from .vigra_util import get_vigra_feature_names, append_vigra_features_to_dataframe

logger = logging.getLogger(__name__)

class VigraSpAccumulator(SpAccumulator):
    def __init__(self, label_img, feature_names):
        self.cleanup() # Initialize members
        self._feature_names = feature_names
        self._vigra_feature_names = get_vigra_feature_names(feature_names)
        self._ndim = label_img.ndim
    
    def cleanup(self):
        self._histogram_range = None
        self._block_vigra_accumulators = []

    def ingest_values_for_block(self, label_block, value_block, block_start, block_stop):
        logger.debug("Computing SP features...")
        block_vigra_acc = self._accumulate_sp_vigra_features( label_block, value_block )
        self._block_vigra_accumulators.append( block_vigra_acc )
    
    def append_merged_sp_features_to_edge_df(self, edge_df):        
        # Merge all the accumulators from each block
        final_sp_acc = self._block_vigra_accumulators[0].createAccumulator()
        for block_vigra_acc in self._block_vigra_accumulators:
            # This is an identity lookup, but it's necessary since vigra will complain 
            # about different maxIds if we call merge() without a lookup 
            axis_to_final_index_array = np.arange( block_vigra_acc.maxRegionLabel()+1, dtype=np.uint32 )
            final_sp_acc.merge( block_vigra_acc, axis_to_final_index_array )


        # Create an almost-empty dataframe to store the sp features
        logger.debug("Saving SP features to DataFrame...")
        index_u32 = pd.Index(np.arange(final_sp_acc.maxRegionLabel()+1), dtype=np.uint32)
        sp_df = pd.DataFrame({ 'sp_id' : np.arange(final_sp_acc.maxRegionLabel()+1, dtype=np.uint32) }, index=index_u32)

        # Add the vigra accumulator results to the SP dataframe
        sp_df = append_vigra_features_to_dataframe(final_sp_acc, sp_df, self._vigra_feature_names, 'sp_')
        
        # Combine SP features and append to the edge_df
        edge_df = self._append_sp_features_onto_edge_features( edge_df, sp_df )
        return edge_df

    def _accumulate_sp_vigra_features(self, label_block, value_block):
        """
        Note: Here we flatten the arrays before passing them to vigra,
              so coordinate-based features won't work.
              This could be easiliy fixed by simply not flattening the arrays,
              but you'll also need to define new suffixes for multi=value features
              (e.g. 'region_axes', etc.), and modify _add_features_to_dataframe() accordingly.
        """
        for feature_name in self._vigra_feature_names:
            for nonsupported_name in ('coord', 'region'):
                # This could be fixed easily (just don't flatten the data)
                # but we should check the performance implications.
                assert nonsupported_name not in feature_name, \
                    "Coordinate-based SP features are not currently supported!"

        histogram_range = self._histogram_range
        if histogram_range is None:
            histogram_range = "globalminmax"
        
        # Convert to float32 if necessary
        value_block = value_block.astype(np.float32, copy=False)
        acc = vigra.analysis.extractRegionFeatures( value_block.reshape((1,-1), order='A'),
                                                    label_block.reshape((1,-1), order='A'),
                                                    features=self._vigra_feature_names,
                                                    histogramRange=histogram_range )

        # If this was the first block we've processed,
        # initialize the histogram_range (if necessary)
        if self._histogram_range is None and set(['quantiles', 'histogram']) & set(self._vigra_feature_names):
            self._histogram_range = [ acc['Global<Minimum >'],
                                      acc['Global<Maximum >'] ]        
        return acc

    def _append_sp_features_onto_edge_features(self, edge_df, sp_df):
        """
        Given a DataFrame with edge features and another DataFrame with superpixel features,
        add columns to the edge_df for each of the specified (superpixel) feature names.
        
        For each sp feature, two columns are added to the output, for the sum and (absolute) difference
        between the feature values for the two superpixels adjacent to the edge.
        (See 'output' feature naming convention notes above for column names.)

        As a special case, the 'count' and 'sum' sp features are normalized first by taking
        their cube roots (or square roots), as indicated in the Multicut paper.
        
        Returns the augmented edge_df.

        Parameters
        ----------
        edge_df
            The dataframe with edge features.
            First columns must be 'sp1', 'sp2'.
            len(edge_df) == self.num_edges
                 
        sp_df
            The dataframe with raw superpixel features.
            First column must be 'sp_id'.
            len(sp_df) == self.num_sp

        generic_vigra_features
            Superpixel feature names without 'sp_' prefix or '_sp1' suffix,
            but possibly with quantile suffix, e.g. '_25'.
            See feature naming convention notes above for details.
        
        ndim
            The dimensionality of the original label volume (an integer).
            Used to normalize the 'count' and 'sum' features.
        """
        # Add two columns to the edge_df for every sp_df column (for sp1 and sp2)
        # note: pd.merge() is like a SQL 'join' operation.
        edge_df = pd.merge( edge_df, sp_df, left_on=['sp1'], right_on=['sp_id'], how='left', copy=False)
        edge_df = pd.merge( edge_df, sp_df, left_on=['sp2'], right_on=['sp_id'], how='left', copy=False, suffixes=('_sp1', '_sp2'))
        del edge_df['sp_id_sp1']
        del edge_df['sp_id_sp2']
    
        # Now create sum/difference columns
        for sp_feature in self._feature_names:
            sp_feature_sum = ( edge_df['sp_' + sp_feature + '_sp1'].values
                             + edge_df['sp_' + sp_feature + '_sp2'].values )
            if sp_feature in ('count', 'sum'):
                # Special case for count
                sp_feature_sum = np.power(sp_feature_sum,
                                          np.float32(1./self._ndim),
                                          out=sp_feature_sum)
            edge_df['sp_' + sp_feature + '_sum'] = sp_feature_sum
    
            sp_feature_difference = ( edge_df['sp_' + sp_feature + '_sp1'].values
                                    - edge_df['sp_' + sp_feature + '_sp2'].values )
            sp_feature_difference = np.abs(sp_feature_difference, out=sp_feature_difference)
            if sp_feature in ('count', 'sum'):
                sp_feature_difference = np.power(sp_feature_difference,
                                                 np.float32(1./self._ndim),
                                                 out=sp_feature_difference)
            edge_df['sp_' + sp_feature + '_difference'] = sp_feature_difference
    
            # Don't need these any more
            del edge_df['sp_' + sp_feature + '_sp1']
            del edge_df['sp_' + sp_feature + '_sp2']
        
        return edge_df
