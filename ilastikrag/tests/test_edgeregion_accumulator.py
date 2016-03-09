import numpy as np
import vigra

from ilastikrag import Rag
from ilastikrag.util import generate_random_voronoi
from ilastikrag.accumulators.edgeregion import EdgeRegionEdgeAccumulator

class TestEdgeRegionEdgeAccumulator(object):

    def test1(self):
        superpixels = generate_random_voronoi((100,200), 200)
        superpixels.axistags = vigra.defaultAxistags('yx')

        feature_names = ['edgeregion_edge_regionradii']

        rag = Rag( superpixels )
        acc = EdgeRegionEdgeAccumulator(rag, feature_names)
        features_df = rag.compute_features(None, feature_names, accumulator_set=[acc])
        radii = features_df[features_df.columns.values[2:]].values
        assert (radii[:,0] >= radii[:,1]).all()
 
        # Transpose superpixels and check again
        # Should match (radii are sorted by magnitude).
        superpixels.axistags = vigra.defaultAxistags('xy')
        rag = Rag( superpixels )
        acc = EdgeRegionEdgeAccumulator(rag, feature_names)
        
        transposed_features_df = rag.compute_features(None, feature_names, accumulator_set=[acc])
        transposed_radii = transposed_features_df[transposed_features_df.columns.values[2:]].values
        
        assert np.isclose(radii, transposed_radii).all()

    def test2(self):
        superpixels = np.zeros((10, 10), dtype=np.uint32)
        superpixels[1:2] = 1
        superpixels = vigra.taggedView(superpixels, 'yx')

        rag = Rag( superpixels )

        feature_names = ['edgeregion_edge_regionradii', 'edgeregion_edge_regionaxes']
        acc = EdgeRegionEdgeAccumulator(rag, feature_names)

        features_df = rag.compute_features(None, feature_names, accumulator_set=[acc])
         
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

    def test_area(self):
        superpixels = generate_random_voronoi((100,200), 200)
        feature_names = ['edgeregion_edge_regionradii', 'edgeregion_edge_area']

        rag = Rag( superpixels )
        features_df = rag.compute_features(None, feature_names)
        
        radii = features_df[['edgeregion_edge_regionradii_0', 'edgeregion_edge_regionradii_1']].values
        assert (radii[:,0] >= radii[:,1]).all()

        areas = features_df[['edgeregion_edge_area']].values
        assert ((radii[:,0] * radii[:,1]) == areas[:,0]).all()

    def test_volume(self):
        superpixels = generate_random_voronoi((100,200), 200)
        feature_names = ['edgeregion_edge_regionradii', 'edgeregion_edge_volume']
        rag = Rag( superpixels )
        
        try:
            rag.compute_features(None, feature_names)
        except AssertionError:
            pass
        except:
            raise
        else:
            assert False, "EdgeRegion accumulator should refuse to compute 'volume' for 2D images."

        superpixels = generate_random_voronoi((25,50,100), 200)
        feature_names = ['edgeregion_edge_regionradii', 'edgeregion_edge_area', 'edgeregion_edge_volume']

        rag = Rag( superpixels )
        features_df = rag.compute_features(None, feature_names)
        
        radii = features_df[['edgeregion_edge_regionradii_0', 'edgeregion_edge_regionradii_1', 'edgeregion_edge_regionradii_2']].values
        assert (radii[:,0] >= radii[:,1]).all() and (radii[:,1] >= radii[:,2]).all()
        
        volumes = features_df[['edgeregion_edge_volume']].values        
        assert ((radii[:,0] * radii[:,1] * radii[:,2]) == volumes[:,0]).all()

        areas = features_df[['edgeregion_edge_area']].values        
        assert ((radii[:,0] * radii[:,1]) == areas[:,0]).all()

    def test_invalid_feature_names(self):
        """
        The Rag should refuse to compute features it doesn't 
        support, not silently omit them.
        """
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        def try_bad_features(feature_names):
            try:
                _ = rag.compute_features(values, feature_names)
            except:
                pass
            else:
                assert False, "Rag should raise an error if the user gives bad feature names!"

        # These aren't supported for a 2D rag
        try_bad_features(['edgeregion_edge_regionradii_2'])
        try_bad_features(['edgeregion_edge_regionaxes_0z'])
        try_bad_features(['edgeregion_edge_regionaxes_1z'])
        try_bad_features(['edgeregion_edge_regionaxes_2x'])
        try_bad_features(['edgeregion_edge_regionaxes_2y'])
        try_bad_features(['edgeregion_edge_regionaxes_2z'])
    
if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)

        
