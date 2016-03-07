import logging
import numpy as np
import pandas as pd

from ilastikrag.accumulators import BaseEdgeAccumulator

logger = logging.getLogger(__name__)

class EdgeRegionEdgeAccumulator(BaseEdgeAccumulator):
    """
    Accumulator for computing region axes and region radii over edge coordinates.
    (The :py:class:`~ilastikrag.accumulators.standard.StandardEdgeAccumulator`
    class does not provide region features.)
    
    We don't use vigra's RegionFeatureAccumulators because we only have
    access to the sparse lists of edge pixels (along each axis).
    Instead, we manually compute the region axes/radii directly from the
    edge coordinate columns.
    
    Supported feature names:

        - edgeregion_edge_regionradii (all of the below)
        - edgeregion_edge_regionradii_0
        - edgeregion_edge_regionradii_1
        - edgeregion_edge_regionradii_2

    ..
    
        - edgeregion_edge_area (radii_0 * radii_1)
        - edgeregion_edge_volume (radii_0 * radii_1 * radii_2)
    ..

        - edgeregion_edge_regionaxes (all of the below)
        - edgeregion_edge_regionaxes_0x
        - edgeregion_edge_regionaxes_0y
        - edgeregion_edge_regionaxes_0z
        - edgeregion_edge_regionaxes_1x
        - edgeregion_edge_regionaxes_1y
        - edgeregion_edge_regionaxes_1z
        - edgeregion_edge_regionaxes_2x
        - edgeregion_edge_regionaxes_2y
        - edgeregion_edge_regionaxes_2z
    """
    ACCUMULATOR_ID = 'edgeregion'

    def __init__(self, rag, feature_names):
        self.cleanup() # Initialize members
        
        label_img = rag.label_img
        self._axisnames = label_img.axistags.keys()
        feature_names = list(feature_names)

        # 'edgeregion_edge_regionradii' is shorthand for "all edge region radii"
        if 'edgeregion_edge_regionradii' in feature_names:
            feature_names.remove('edgeregion_edge_regionradii')
            for component_index in range(label_img.ndim):
                feature_names.append( 'edgeregion_edge_regionradii_{}'.format( component_index ) )            
        
        # 'edgeregion_edge_regionaxes' is shorthand for "all edge region axes"
        if 'edgeregion_edge_regionaxes' in feature_names:
            feature_names.remove('edgeregion_edge_regionaxes')
            for component_index in range(label_img.ndim):
                for axisname in map( lambda k: 'xyz'[k], range(label_img.ndim) ):
                    feature_names.append( 'edgeregion_edge_regionaxes_{}{}'.format( component_index, axisname ) )            
        
        self._feature_names = feature_names
        self._rag = rag
    
    def cleanup(self):
        self._final_df = None

    def ingest_edges_for_block(self, axial_edge_dfs, block_start, block_stop):
        assert self._final_df is None, \
            "This accumulator doesn't support block-wise merging (yet).\n"\
            "You can only process a volume as a single block"

        # Concatenate edges from all axes into one big DataFrame
        coords_df = pd.concat(axial_edge_dfs)[['sp1', 'sp2'] + self._axisnames]
        
        # Create a new DataFrame to store the results
        final_df = pd.DataFrame(self._rag.edge_ids, columns=['sp1', 'sp2'])
        
        num_edges = len(final_df)
        ndim = len(self._axisnames)
        covariance_matrices_array = np.zeros( (num_edges, ndim, ndim), dtype=np.float32 )

        group_index = [-1]
        def write_covariance_matrix(group_df):
            """
            Computes the covariance matrix of the given group,
            and writes it into the pre-existing covariance_matrices_array.
            """
            # There's one 'gotcha' to watch out for here:
            # GroupBy.apply() calls this function *twice* for the first group.
            # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.GroupBy.apply.html
            if group_index[0] < 0:
                group_index[0] += 1
                return None
            
            # Compute covariance.
            # Apparently computing it manually is much faster than group_df.cov()
            group_vals = group_df.values.astype(np.float32, copy=False)
            group_vals -= group_vals.mean(axis=0)
            matrix = group_vals.transpose().dot(group_vals)
            matrix[:] /= len(group_df)
            
            # Store.
            covariance_matrices_array[group_index[0]] = matrix
            group_index[0] += 1
            
            # We don't need to return anything;
            # we're using this function only for its side effects.
            return None

        # Compute/store covariance matrices
        grouper = coords_df.groupby(['sp1', 'sp2'], sort=True, group_keys=False)
        grouper = grouper[self._axisnames]
        grouper.apply(write_covariance_matrix) # Used for its side-effects only

        # Eigensystems
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrices_array)
        assert eigenvalues.shape == (num_edges, ndim)
        assert eigenvectors.shape == (num_edges, ndim, ndim)

        # eigh() returns in *ascending* order, but we want descending
        eigenvalues = eigenvalues[:, ::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # eigh() returns eigenvectors in *columns*, but we want rows
        eigenvectors = eigenvectors.transpose(0,2,1)        

        # Apparently eigh() can return tiny negative eigenvalues sometimes
        eigenvalues[eigenvalues < 0.0] = 0.0
        radii = np.sqrt(eigenvalues, out=eigenvalues)

        # Copy axes into final_df, in the same order the user asked for.
        for feature_name in self._feature_names:
            if feature_name.startswith('edgeregion_edge_regionradii'):
                region_axis_index = int(feature_name[-1])
                final_df[feature_name] = radii[:, region_axis_index]
            elif feature_name.startswith('edgeregion_edge_regionaxes'):
                region_axis_index = int(feature_name[-2])
                coord_index = self._axisnames.index(feature_name[-1])
                final_df[feature_name] = eigenvectors[:, region_axis_index, coord_index]
            elif feature_name == 'edgeregion_edge_area':
                final_df[feature_name] = np.prod(radii[:, :2], axis=1)
            elif feature_name == 'edgeregion_edge_volume':
                assert radii.shape[-1] == 3, "Can't ask for edge volume with 2D images."
                final_df[feature_name] = np.prod(radii, axis=1)
            else:
                assert False, "Unknown feature: ".format( feature_name )

        self._final_df = final_df
    
    def append_merged_edge_features_to_df(self, edge_df):
        # This accumulator doesn't support blockwise processing and merging,
        # so just merge our one-and-only block results into the edge_df.
        return pd.merge(edge_df, self._final_df, on=['sp1', 'sp2'], how='left', copy=False)
