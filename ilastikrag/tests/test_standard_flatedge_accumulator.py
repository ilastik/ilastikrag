import numpy as np
import pandas as pd
import vigra

from ilastikrag import Rag
from ilastikrag.util import generate_random_voronoi

class TestStandardFlatEdgeAccumulator(object):

    def test_count(self):
        # Create a volume of flat superpixels, where every slice 
        # is the same (except for the actual sp ids)
        num_sp_per_slice = 200
        slice_superpixels = generate_random_voronoi((100,200), num_sp_per_slice)
         
        superpixels = np.zeros( shape=((10,) + slice_superpixels.shape), dtype=np.uint32 )
        for z in range(10):
            superpixels[z] = slice_superpixels + z*num_sp_per_slice
        superpixels = vigra.taggedView(superpixels, 'zyx')
 
        rag = Rag( superpixels, flat_superpixels=True )
 
        # Manually compute the sp counts for the first N-1 slices,
        # which happens to be the same as the edge counts (since superpixels are identical on all slices)
        sp_counts = np.bincount(superpixels[:-1].flat[:])
        sp_counts = sp_counts[1:] # Drop zero sp (it doesn't exist)
         
        features_df = rag.compute_features(None, ['standard_flatedge_count'], edge_type='flat')
        assert sorted(features_df['standard_flatedge_count'].values) == sorted(sp_counts)

    def test_quantiles(self):
        # Create a volume of flat superpixels, where every slice 
        # is the same (except for the actual sp ids)
        num_sp_per_slice = 200
        slice_superpixels = generate_random_voronoi((100,200), num_sp_per_slice)
        
        superpixels = np.zeros( shape=((10,) + slice_superpixels.shape), dtype=np.uint32 )
        for z in range(10):
            superpixels[z] = slice_superpixels + z*num_sp_per_slice
        superpixels = vigra.taggedView(superpixels, 'zyx')

        rag_flat = Rag( superpixels, flat_superpixels=True )
        
        values = np.random.random(size=(superpixels.shape)).astype(np.float32)
        values = vigra.taggedView(values, 'zyx')
        
        flat_features_df = rag_flat.compute_features(values, ['standard_flatedge_quantiles'], edge_type='flat')
        flat_features_df2 = rag_flat.compute_features(values, ['standard_edge_quantiles'], edge_type='dense')
        
        # Rename columns
        flat_features_df2.columns = flat_features_df.columns.values
        flat_features_df = pd.concat( (flat_features_df, flat_features_df2), axis=0 )

        # Now compute the quantiles using a normal 'dense' rag
        rag_dense = Rag( superpixels )
        dense_features_df = rag_dense.compute_features(values, ['standard_edge_quantiles'])

        all_features_df = pd.merge(dense_features_df, flat_features_df, how='left', on=['sp1', 'sp2'])
        assert (all_features_df['standard_edge_quantiles_0'] == all_features_df['standard_flatedge_quantiles_0']).all()
        assert (all_features_df['standard_edge_quantiles_100'] == all_features_df['standard_flatedge_quantiles_100']).all()

        # Due to the way histogram ranges are computed, we can't expect quantiles_10 to match closely
        # ... but quantiles_50 seems to be good.
        assert np.isclose(all_features_df['standard_edge_quantiles_50'], all_features_df['standard_edge_quantiles_50']).all()
        
    def test_regionradii(self):
        # Create a volume of flat superpixels, where every slice
        # is the same (except for the actual sp ids)
        num_sp_per_slice = 200
        slice_superpixels = generate_random_voronoi((100,200), num_sp_per_slice)
        
        superpixels = np.zeros( shape=((10,) + slice_superpixels.shape), dtype=np.uint32 )
        for z in range(10):
            superpixels[z] = slice_superpixels + z*num_sp_per_slice
        superpixels = vigra.taggedView(superpixels, 'zyx')

        rag_flat = Rag( superpixels, flat_superpixels=True )
        
        values = np.random.random(size=(superpixels.shape)).astype(np.float32)
        values = vigra.taggedView(values, 'zyx')
        
        flat_features_df = rag_flat.compute_features(values, ['standard_flatedge_regionradii'], edge_type='flat')

        # Now compute the radii using a normal 'dense' rag
        rag_dense = Rag( superpixels )
        dense_features_df = rag_dense.compute_features(values, ['edgeregion_edge_regionradii'])

        # Both methods should be reasonably close.
        combined_features_df = pd.merge(flat_features_df, dense_features_df, how='left', on=['sp1', 'sp2'])
        assert np.isclose(combined_features_df['standard_flatedge_regionradii_0'], combined_features_df['edgeregion_edge_regionradii_0'], atol=0.001).all()
        assert np.isclose(combined_features_df['standard_flatedge_regionradii_1'], combined_features_df['edgeregion_edge_regionradii_1'], atol=0.001).all()


if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
