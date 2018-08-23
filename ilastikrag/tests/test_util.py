import tempfile
import shutil

import numpy as np
import h5py

from ilastikrag.rag import Rag
from ilastikrag.util import label_vol_mapping, generate_random_voronoi, dataframe_to_hdf5, dataframe_from_hdf5

def test_label_vol_mapping():
    # 1 2
    # 3 4    
    vol1 = np.zeros((20,20), dtype=np.uint8)
    vol1[ 0:10,  0:10] = 1
    vol1[ 0:10, 10:20] = 2
    vol1[10:20,  0:10] = 3
    vol1[10:20, 10:20] = 4

    # 2 3
    # 4 5
    vol2 = vol1.copy() + 1
    
    assert (label_vol_mapping(vol1, vol2) == [0,2,3,4,5]).all()
    assert (label_vol_mapping(vol1[3:], vol2[:-3]) == [0,2,3,4,5]).all()
    assert (label_vol_mapping(vol1[6:], vol2[:-6]) == [0,2,3,2,3]).all()
    
    # 7 7
    # 4 5
    vol2[( vol2 == 2 ).nonzero()] = 7
    vol2[( vol2 == 3 ).nonzero()] = 7

    assert (label_vol_mapping(vol1, vol2) == [0,7,7,4,5]).all()

def test_features_df_serialization():
    superpixels = generate_random_voronoi((100,200), 200)
    rag = Rag( superpixels )

    # For simplicity, just make values identical to superpixels
    values = superpixels.astype(np.float32)

    feature_names = ['standard_edge_mean', 'standard_edge_minimum', 'standard_edge_maximum']
    features_df = rag.compute_features(values, feature_names)

    tmpdir = tempfile.mkdtemp()
    try:
        with h5py.File( tmpdir + '/' + 'test_dataframe.h5', 'w' ) as f:
            group = f.create_group('test_dataframe')
            dataframe_to_hdf5( group, features_df )
        
        with h5py.File( tmpdir + '/' + 'test_dataframe.h5', 'r' ) as f:
            group = f['test_dataframe']
            readback_features_df = dataframe_from_hdf5( group )
        
        assert (readback_features_df.columns.values == features_df.columns.values).all()
        assert (readback_features_df.values.shape == features_df.values.shape)
        assert (readback_features_df.values == features_df.values).all()

    finally:
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    import os
    import pytest
    module = os.path.split(__file__)[1][:-3]
    pytest.main(['-s', '--tb=native', '--pyargs', f'ilastikrag.tests.{module}'])
