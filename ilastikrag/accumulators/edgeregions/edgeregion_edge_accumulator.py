import logging
import numpy as np
import pandas as pd

from ilastikrag.accumulators import BaseEdgeAccumulator

logger = logging.getLogger(__name__)

class EdgeRegionEdgeAccumulator(BaseEdgeAccumulator):
    """
    Accumulator for computing region axes and regionradii of edge pixels.
    
    Supported feature names:

        - edgeregion_edge_regionradii (all of the below)
        - edgeregion_edge_regionradii_0
        - edgeregion_edge_regionradii_1
        - edgeregion_edge_regionradii_2

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

    def __init__(self, label_img, feature_names):
        self.cleanup() # Initialize members
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
    
    def cleanup(self):
        self._num_blocks = 0
        self._final_df = None

    def ingest_edges_for_block(self, axial_edge_dfs, block_start, block_stop):
        assert self._num_blocks == 0, \
            "This accumulator doesn't support block-wise merging (yet).\n"\
            "You can only process a volume as a single block"
        self._num_blocks += 1
        
        # Compute edge_centroids
        coords_df = pd.concat(axial_edge_dfs)[['sp1', 'sp2'] + self._axisnames]
        centroids_df = coords_df.groupby(['sp1', 'sp2'], as_index=False)[self._axisnames].mean()

        # Rename z,y,x columns        
        centroid_colnames = map( lambda name: name + '_centroid', self._axisnames )
        centroids_df.columns = ['sp1', 'sp2'] + centroid_colnames
        
        # Add centroid columns. Columns are:
        # ['sp1', 'sp2', 'z', 'y', 'x', 'z_centroid', 'y_centroid', 'x_centroid']
        coords_df = pd.merge(coords_df, centroids_df, on=['sp1', 'sp2'], how='left', copy=False)
        
        # Centralize: (x - x_centroid), etc.
        centralized_coords = coords_df[self._axisnames].values - coords_df[centroid_colnames].values.astype(np.float32)
        centralized_coords_df = pd.DataFrame(centralized_coords, columns=self._axisnames)
        centralized_coords_df = coords_df[['sp1', 'sp2']].join(centralized_coords_df)        
        
        def column_covariance(group_df):
            # Compute the covariance matrix of the columns in group_df,
            # and then add the matrix to a DataFrame as a single element (dtype=object)
            # Note: The columns of group_df should already be centralized; i.e. they have mean=0.0 
            group_vals = group_df.values.astype(np.float32)
            matrix = group_vals.transpose().dot(group_vals)
            matrix[:] /= len(group_df)
            df = pd.DataFrame({ 'covariance_matrix': [[matrix]] })
            assert df.shape == (1,1)
            return df

        # Compute covariance matrices
        # The 'groupdot' column contains a matrix in each element (dtype=object).
        covariance_matrices_df = centralized_coords_df.groupby(['sp1', 'sp2'], as_index=False)[self._axisnames].apply(column_covariance)        
        covariance_matrices_array = np.concatenate(covariance_matrices_df.values[:,0])
        
        num_edges = len(centroids_df)
        ndim = len(self._axisnames)
        assert covariance_matrices_array.ndim == 3
        assert covariance_matrices_array.shape == (num_edges, ndim, ndim)
        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrices_array)
        assert eigenvalues.shape == (num_edges, ndim)
        assert eigenvectors.shape == (num_edges, ndim, ndim)

        # Apparently eigh can return small negative eigenvalues sometimes
        eigenvalues[eigenvalues < 0.0] = 0.0
        
        # np.eigh() returns in *ascending* order, but we want descending
        eigenvalues = eigenvalues[:, ::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # np.eigh() returns eigenvectors in *columns*, but we want rows
        eigenvectors = eigenvectors.transpose(0,2,1)        

        radii = np.sqrt(eigenvalues, out=eigenvalues)

        # Copy axes into final_df, in the same order the user asked for.
        final_df = pd.DataFrame(centroids_df[['sp1', 'sp2']])
        for feature_name in self._feature_names:
            if feature_name.startswith('edgeregion_edge_regionradii'):
                region_axis_index = int(feature_name[-1])
                final_df[feature_name] = radii[:, region_axis_index]
            elif feature_name.startswith('edgeregion_edge_regionaxes'):
                region_axis_index = int(feature_name[-2])
                coord_index = self._axisnames.index(feature_name[-1])
                final_df[feature_name] = eigenvectors[:, region_axis_index, coord_index]

        self._final_df = final_df
    
    def append_merged_edge_features_to_df(self, edge_df):
        return pd.merge(edge_df, self._final_df, on=['sp1', 'sp2'], how='left', copy=False)
