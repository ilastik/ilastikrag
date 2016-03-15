import numpy as np
import vigra

from ilastikrag import Rag
from ilastikrag.util import generate_random_voronoi

class TestSimilarityFlatEdgeAccumulator(object):

    def test_correlation(self):
        # Create a volume of flat superpixels, where every slice 
        # is the same (except for the actual sp ids)
        num_sp_per_slice = 200
        slice_superpixels = generate_random_voronoi((100,200), num_sp_per_slice)
        
        superpixels = np.zeros( shape=((10,) + slice_superpixels.shape), dtype=np.uint32 )
        for z in range(10):
            superpixels[z] = slice_superpixels + z*num_sp_per_slice
        superpixels = vigra.taggedView(superpixels, 'zyx')

        rag = Rag( superpixels, flat_superpixels=True )

        # For simplicity, just make values identical to the first slice
        values = np.zeros_like(superpixels, dtype=np.float32)
        values[:] = slice_superpixels[None]

        features_df = rag.compute_features(values, ['similarity_flatedge_correlation'], edge_group='z')
        assert (features_df['similarity_flatedge_correlation'].values == 1.0).all()

        # Now add a little noise from one slice to the next.
        for z in range(10):
            if z == 0:
                continue
            if z == 1:
                values[z] += 0.001*np.random.random(size=(values[z].shape))
            else:
                values[z] += 1.0001*np.abs(values[z-1] - values[z-2])

        features_df = rag.compute_features(values, ['similarity_flatedge_correlation'], edge_group='z')        
        assert (features_df['similarity_flatedge_correlation'].values >= 0.7).all()
        assert (features_df['similarity_flatedge_correlation'].values <= 1.0).all()
        
        # Now use just noise
        values = np.random.random(size=values.shape).astype(np.float32)
        values = vigra.taggedView(values, 'zyx')
        features_df = rag.compute_features(values, ['similarity_flatedge_correlation'], edge_group='z')
        assert (features_df['similarity_flatedge_correlation'].values <= 1.0).all()
        assert (features_df['similarity_flatedge_correlation'].values >= -1.0).all()
        

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
