import os
import tempfile
import itertools

import numpy as np
import pandas as pd
import vigra

from ilastikrag import Rag
from ilastikrag.util import generate_random_voronoi

class TestRag(object):
    
    def test_construction(self):
        superpixels = generate_random_voronoi((100,200), 200)
        
        rag = Rag( superpixels )
        assert rag.num_sp == 200, "num_sp was: {}".format(rag.num_sp)
        assert rag.max_sp == 200
        assert (rag.sp_ids == np.arange(1,201)).all()
        assert rag.sp_ids.dtype == np.uint32
        assert isinstance(rag.max_sp, np.uint32)
        assert rag.edge_ids.dtype == np.uint32
        assert (rag.unique_edge_tables['yx'][['sp1', 'sp2', 'edge_label']].dtypes == np.uint32).all()
        for _axiskey, df in rag.dense_edge_tables.items():
            assert df['sp1'].dtype == df['sp2'].dtype == np.uint32
            assert df['forwardness'].dtype == bool
            assert df['edge_label'].dtype == np.uint32
            
            # The coordinate dtype can be uint16 or uint32,
            # depending on the size of the size of the image 
            assert df['y'].dtype == df['x'].dtype == np.uint16

        # Just check some basic invariants of the edge_ids
        assert rag.edge_ids.shape == (rag.num_edges, 2)

        # For all edges (u,v): u < v
        edge_ids_copy = rag.edge_ids.copy()
        edge_ids_copy.sort(axis=1)
        assert (rag.edge_ids == edge_ids_copy).all()

        # edge_ids should be sorted by u, then v.
        edge_df = pd.DataFrame(edge_ids_copy, columns=['sp1', 'sp2'])
        edge_df.sort_values(['sp1', 'sp2'], inplace=True)
        assert (rag.edge_ids == edge_df.values).all()

        # We're just using default features, so the supported features should match.
        default_features = [acc_cls.supported_features(rag) for acc_cls in Rag.DEFAULT_ACCUMULATOR_CLASSES.values()]
        default_features = itertools.chain(*default_features)
        assert set(rag.supported_features()) == set( default_features )

    def test_superpixel_edges_only_in_y_direction(self):
        superpixels = np.zeros((5, 6), dtype="uint32")
        superpixels[0:2, ...] = 1
        superpixels[2:, ...] = 2
        superpixels = vigra.taggedView(superpixels, axistags="yx")
        rag = Rag(superpixels)

        dense_edge_tables = rag.dense_edge_tables
        assert len(dense_edge_tables["x"]) == 0
        assert isinstance(dense_edge_tables["x"].index, pd.Int64Index)

        assert len(dense_edge_tables["y"]) == 6
        assert isinstance(dense_edge_tables["y"].index, pd.Int64Index)

        for column in dense_edge_tables["y"].columns:
            assert dense_edge_tables["x"][column].dtype == dense_edge_tables["y"][column].dtype

    def test_edge_decisions_from_groundtruth(self):
        # 1 2
        # 3 4
        vol1 = np.zeros((20,20), dtype=np.uint32)
        vol1[ 0:10,  0:10] = 1
        vol1[ 0:10, 10:20] = 2
        vol1[10:20,  0:10] = 3
        vol1[10:20, 10:20] = 4
        
        vol1 = vigra.taggedView(vol1, 'yx')
        rag = Rag(vol1)
    
        # 2 3
        # 4 5
        vol2 = vol1.copy() + 1

        decisions = rag.edge_decisions_from_groundtruth(vol2)
        assert decisions.all()

        # 7 7
        # 4 5
        vol2[( vol2 == 2 ).nonzero()] = 7
        vol2[( vol2 == 3 ).nonzero()] = 7
        
        decision_dict = rag.edge_decisions_from_groundtruth(vol2, asdict=True)
        assert decision_dict[(1,2)] == False
        assert decision_dict[(1,3)] == True
        assert decision_dict[(2,4)] == True
        assert decision_dict[(3,4)] == True

    def test_naive_segmentation_from_edge_decisions(self):
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )
        
        # The 'groundtruth' is just divided into quadrants
        groundtruth = np.zeros_like(superpixels)
        groundtruth[0:50,   0:100] = 1
        groundtruth[50:100, 0:100] = 2
        groundtruth[0:50,   100:200] = 3
        groundtruth[50:100, 100:200] = 4

        decisions = rag.edge_decisions_from_groundtruth(groundtruth)
        segmentation = rag.naive_segmentation_from_edge_decisions(decisions)
        
        # We don't know where the exact boundary is, but pixels 
        # near the corners should definitely be homogenous
        assert (segmentation[:20,   :20] == segmentation[0,  0]).all()
        assert (segmentation[:20,  -20:] == segmentation[0, -1]).all()
        assert (segmentation[-20:,  :20] == segmentation[-1, 0]).all()
        assert (segmentation[-20:, -20:] == segmentation[-1,-1]).all()

    def test_serialization_with_labels(self):
        """
        Serialize the rag and labels to hdf5,
        then deserialize it and make sure nothing was lost.
        """
        import h5py
 
        superpixels = generate_random_voronoi((100,200), 200)
        original_rag = Rag( superpixels )
 
        tmp_dir = tempfile.mkdtemp()
        filepath = os.path.join(tmp_dir, 'test_rag.h5')
        rag_groupname = 'saved_rag'
 
        # Serialize with labels       
        with h5py.File(filepath, 'w') as f:
            rag_group = f.create_group(rag_groupname)
            original_rag.serialize_hdf5(rag_group, store_labels=True)
 
        # Deserialize
        with h5py.File(filepath, 'r') as f:
            rag_group = f[rag_groupname]
            deserialized_rag = Rag.deserialize_hdf5(rag_group)
 
        assert deserialized_rag.label_img.dtype == original_rag.label_img.dtype
        assert deserialized_rag.label_img.shape == original_rag.label_img.shape
        assert deserialized_rag.label_img.axistags == original_rag.label_img.axistags
        assert (deserialized_rag.label_img == original_rag.label_img).all()        
 
        assert (deserialized_rag.sp_ids == original_rag.sp_ids).all()
        assert deserialized_rag.max_sp == original_rag.max_sp
        assert deserialized_rag.num_sp == original_rag.num_sp
        assert deserialized_rag.num_edges == original_rag.num_edges
        assert (deserialized_rag.edge_ids == original_rag.edge_ids).all()
 
        # Check some features
        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)
        feature_names = ['standard_edge_mean', 'standard_sp_count']
        features_df_original = original_rag.compute_features(values, feature_names)
        features_df_deserialized = deserialized_rag.compute_features(values, feature_names)
        assert (features_df_original.values == features_df_deserialized.values).all()
 
    def test_serialization_without_labels(self):
        """
        Users can opt to serialize the Rag without serializing the labels,
        but then they can't use superpixel features on the deserialized Rag.
        """
        import h5py
 
        superpixels = generate_random_voronoi((100,200), 200)
        original_rag = Rag( superpixels )
 
        tmp_dir = tempfile.mkdtemp()
        filepath = os.path.join(tmp_dir, 'test_rag.h5')
        rag_groupname = 'saved_rag'
 
        # Serialize with labels       
        with h5py.File(filepath, 'w') as f:
            rag_group = f.create_group(rag_groupname)
            original_rag.serialize_hdf5(rag_group, store_labels=False) # Don't store
 
        # Deserialize
        with h5py.File(filepath, 'r') as f:
            rag_group = f[rag_groupname]
            deserialized_rag = Rag.deserialize_hdf5(rag_group)
 
        assert deserialized_rag.label_img.dtype == original_rag.label_img.dtype
        assert deserialized_rag.label_img.shape == original_rag.label_img.shape
        assert deserialized_rag.label_img.axistags == original_rag.label_img.axistags
        #assert (deserialized_rag.label_img == original_rag.label_img).all() # not stored
 
        assert (deserialized_rag.sp_ids == original_rag.sp_ids).all()
        assert deserialized_rag.max_sp == original_rag.max_sp
        assert deserialized_rag.num_sp == original_rag.num_sp
        assert deserialized_rag.num_edges == original_rag.num_edges
        assert (deserialized_rag.edge_ids == original_rag.edge_ids).all()
 
        # Check some features
        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)
        feature_names = ['standard_edge_mean', 'standard_edge_count']
        features_df_original = original_rag.compute_features(values, feature_names)
        features_df_deserialized = deserialized_rag.compute_features(values, feature_names)
        assert (features_df_original.values == features_df_deserialized.values).all()
 
        try:
            deserialized_rag.compute_features(values, ['standard_sp_count'])
        except NotImplementedError:
            pass
        except:
            raise
        else:
            assert False, "Shouldn't be able to use superpixels if labels weren't serialized/deserialized!"
 
    def test_serialization_with_external_labels(self):
        """
        Users can opt to serialize the Rag without serializing the labels,
        but then they can't use superpixel features on the deserialized Rag.
         
        When deserializing, they can provide the labels from an external source,
        as tested here.
        """
        import h5py
 
        superpixels = generate_random_voronoi((100,200), 200)
        original_rag = Rag( superpixels )
 
        tmp_dir = tempfile.mkdtemp()
        filepath = os.path.join(tmp_dir, 'test_rag.h5')
        rag_groupname = 'saved_rag'
 
        # Serialize with labels       
        with h5py.File(filepath, 'w') as f:
            rag_group = f.create_group(rag_groupname)
            original_rag.serialize_hdf5(rag_group, store_labels=False)
 
        # Deserialize
        with h5py.File(filepath, 'r') as f:
            rag_group = f[rag_groupname]
            deserialized_rag = Rag.deserialize_hdf5(rag_group, label_img=superpixels) # Provide labels explicitly
 
        assert deserialized_rag.label_img.dtype == original_rag.label_img.dtype
        assert deserialized_rag.label_img.shape == original_rag.label_img.shape
        assert deserialized_rag.label_img.axistags == original_rag.label_img.axistags
        assert (deserialized_rag.label_img == original_rag.label_img).all()
 
        assert (deserialized_rag.sp_ids == original_rag.sp_ids).all()
        assert deserialized_rag.max_sp == original_rag.max_sp
        assert deserialized_rag.num_sp == original_rag.num_sp
        assert deserialized_rag.num_edges == original_rag.num_edges
        assert (deserialized_rag.edge_ids == original_rag.edge_ids).all()
 
        # Check some features
        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)
        feature_names = ['standard_edge_mean', 'standard_sp_count']
        features_df_original = original_rag.compute_features(values, feature_names)
        features_df_deserialized = deserialized_rag.compute_features(values, feature_names)
        
        assert (features_df_original.values == features_df_deserialized.values).all()

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

        try_bad_features(['standard_edddddge_count'])
        try_bad_features(['standard_sssp_count'])
        try_bad_features(['ssssstandard_sp_count'])
        try_bad_features(['ssssstandard_edge_count'])
        try_bad_features(['standard_edge_countttt'])
        try_bad_features(['standard_sp_countttt'])
        

if __name__ == "__main__":
    import os
    import pytest
    module = os.path.split(__file__)[1][:-3]
    pytest.main(['-s', '--tb=native', '--pyargs', f'ilastikrag.tests.{module}'])
