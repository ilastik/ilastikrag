import numpy as np
import vigra

from ilastikrag import Rag
from ilastikrag.util import generate_random_voronoi
from ilastikrag.accumulators.edgeregions import EdgeRegionEdgeAccumulator

class TestEdgeRegionEdgeAccumulator(object):

    if EdgeRegionEdgeAccumulator not in Rag.ACCUMULATOR_CLASSES:
        Rag.ACCUMULATOR_CLASSES += [EdgeRegionEdgeAccumulator]
    
    def test1(self):
         
        superpixels = generate_random_voronoi((100,200), 200)
        superpixels.axistags = vigra.defaultAxistags('yx')
        values = np.zeros_like(superpixels, dtype=np.float32)

        rag = Rag( superpixels )
        features_df = rag.compute_features(values, ['edgeregion_edge_regionradii'])
        radii = features_df[features_df.columns.values[2:]].values
 
        # Transpose superpixels and check again
        # Should match (radii are sorted by magnitude).
        superpixels.axistags = vigra.defaultAxistags('xy')
        rag = Rag( superpixels ) 
        transposed_features_df = rag.compute_features(values, ['edgeregion_edge_regionradii'])        
        transposed_radii = transposed_features_df[transposed_features_df.columns.values[2:]].values
        
        assert np.isclose(radii, transposed_radii).all()

    def test2(self):
        superpixels = np.zeros((10, 10), dtype=np.uint32)
        superpixels[1:2] = 1
        superpixels = vigra.taggedView(superpixels, 'yx')
        rag = Rag( superpixels )
         
        values = np.zeros_like(superpixels, dtype=np.float32)        
        features_df = rag.compute_features(values, ['edgeregion_edge_regionradii', 'edgeregion_edge_regionaxes'])
         
        # Just 1 edge in this rag (0 -> 1)
        assert len(features_df) == 1
        assert (features_df[['sp1', 'sp2']].values == [0,1]).all()

        # Manually compute the radius 
        x_coords = np.arange(0, 10)
        x_coord_mean = x_coords.mean()
        centralized_x_coords = x_coords - x_coord_mean
        x_coord_variance = centralized_x_coords.dot(centralized_x_coords)/len(x_coords)
        x_radius = np.sqrt(x_coord_variance).astype(np.float32)
         
        assert features_df['edgeregion_edge_regionradii_0'].values[0] == x_radius        
 
        # Eigenvectors are just parallel to the axes
        assert features_df['edgeregion_edge_regionaxes_0x'].values[0] == 1.0
        assert features_df['edgeregion_edge_regionaxes_0y'].values[0] == 0.0
        
        assert features_df['edgeregion_edge_regionaxes_1x'].values[0] == 0.0
        assert features_df['edgeregion_edge_regionaxes_1y'].values[0] == 1.0


if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)

        
