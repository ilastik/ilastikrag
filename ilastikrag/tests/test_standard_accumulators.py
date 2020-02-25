from collections import OrderedDict
import numpy as np
import vigra

from ilastikrag import Rag
from ilastikrag.util import generate_random_voronoi

class TestStandardAccumulators(object):
    
    def test_sp_features_no_histogram(self):
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        # Manually compute the sp counts
        sp_counts = np.bincount(superpixels.flat[:])

        # COUNT
        features_df = rag.compute_features(values, ['standard_sp_count'])
        assert len(features_df) == len(rag.edge_ids)
        assert (features_df.columns.values == ['sp1', 'sp2', 'standard_sp_count_sum', 'standard_sp_count_difference']).all()
        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()
        dtypes = { colname: series.dtype for colname, series in features_df.iteritems() }
        assert all(dtype != np.float64 for dtype in dtypes.values()), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        # sp count features are normalized, consistent with the multicut paper.
        for _index, sp1, sp2, sp_count_sum, sp_count_difference in features_df.itertuples():
            np.testing.assert_almost_equal(
                sp_count_sum,
                np.power(sp_counts[sp1] + sp_counts[sp2], 1./superpixels.ndim).astype(np.float32),
                decimal=6)
            np.testing.assert_almost_equal(
                sp_count_difference,
                np.power(np.abs(sp_counts[sp1] - sp_counts[sp2]), 1./superpixels.ndim).astype(np.float32),
                decimal=6)

        # SUM
        features_df = rag.compute_features(values, ['standard_sp_sum'])
        assert len(features_df) == len(rag.edge_ids)
        assert (features_df.columns.values == ['sp1', 'sp2', 'standard_sp_sum_sum', 'standard_sp_sum_difference']).all()
        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # sp sum features ought to be normalized, too...
        for _index, sp1, sp2, sp_sum_sum, sp_sum_difference in features_df.itertuples():
            np.testing.assert_almost_equal(
                sp_sum_sum,
                np.power(sp1*sp_counts[sp1] + sp2*sp_counts[sp2], 1./superpixels.ndim).astype(np.float32),
                decimal=6)
            np.testing.assert_almost_equal(
                sp_sum_difference,
                np.power(np.abs(sp1*sp_counts[sp1] - sp2*sp_counts[sp2]), 1./superpixels.ndim).astype(np.float32),
                decimal=6)

        # MEAN
        features_df = rag.compute_features(values, ['standard_sp_mean'])
        assert len(features_df) == len(rag.edge_ids)
        assert (features_df.columns.values == ['sp1', 'sp2', 'standard_sp_mean_sum', 'standard_sp_mean_difference']).all()
        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # No normalization for other features...
        # Should there be?
        for _index, sp1, sp2, sp_mean_sum, sp_mean_difference in features_df.itertuples():
            assert sp_mean_sum == sp1 + sp2
            assert sp_mean_difference == np.abs(np.float32(sp1) - sp2)

    def test_sp_features_with_histogram(self):
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        # Manually compute the sp counts
        sp_counts = np.bincount(superpixels.flat[:])

        # COUNT
        features_df = rag.compute_features(values, ['standard_sp_count', 'standard_sp_quantiles_25', 'standard_sp_quantiles_75'])
        assert len(features_df) == len(rag.edge_ids)
        assert (features_df.columns.values == ['sp1', 'sp2',
                                               'standard_sp_count_sum',
                                               'standard_sp_count_difference',
                                               'standard_sp_quantiles_25_sum',
                                               'standard_sp_quantiles_25_difference',
                                               'standard_sp_quantiles_75_sum',
                                               'standard_sp_quantiles_75_difference']).all()

        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # Check dtypes (pandas makes it too easy to get this wrong).
        dtypes = { colname: series.dtype for colname, series in features_df.iteritems() }
        assert all(dtype != np.float64 for dtype in dtypes.values()), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        # sp count features are normalized, consistent with the multicut paper.
        for _index, sp1, sp2, \
            sp_count_sum, sp_count_difference, \
            sp_quantiles_25_sum, sp_quantiles_25_difference, \
            sp_quantiles_75_sum, sp_quantiles_75_difference in features_df.itertuples():
            
            assert type(sp_quantiles_25_sum) in [np.float32, float], f"got {type(sp_quantiles_25_sum)} instead of np.float32"
            assert type(sp_quantiles_75_sum) in [np.float32, float]
            assert type(sp_quantiles_25_difference) in [np.float32, float]
            assert type(sp_quantiles_75_difference) in [np.float32, float]
            
            assert sp_count_sum == np.power(sp_counts[sp1] + sp_counts[sp2], 1./superpixels.ndim).astype(np.float32)
            assert sp_count_difference == np.power(np.abs(sp_counts[sp1] - sp_counts[sp2]), 1./superpixels.ndim).astype(np.float32)
            assert sp_quantiles_25_sum == float(sp1 + sp2), \
                "{} != {}".format( sp_quantiles_25_sum, float(sp1 + sp2) )
            assert sp_quantiles_75_sum == float(sp1 + sp2)
            assert sp_quantiles_25_difference == abs(float(sp1) - sp2)
            assert sp_quantiles_75_difference == abs(float(sp1) - sp2)

    def test_sp_region_features(self):
        # Create a superpixel for each y-column
        superpixels = np.zeros( (100, 200), dtype=np.uint32 )
        superpixels[:] = np.arange(200)[None, :]
        superpixels = vigra.taggedView(superpixels, 'yx')
        
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        feature_names = ['standard_sp_regionradii_0',
                         'standard_sp_regionradii_1',
                         'standard_sp_regionaxes_0x',
                         'standard_sp_regionaxes_0y',
                         'standard_sp_regionaxes_1x',
                         'standard_sp_regionaxes_1y',
                         ]

        output_columns = ['sp1', 'sp2',
                          'standard_sp_regionradii_0_sum',
                          'standard_sp_regionradii_0_difference',
                          'standard_sp_regionradii_1_sum',
                          'standard_sp_regionradii_1_difference',
                        
                          'standard_sp_regionaxes_0x_sum',
                          'standard_sp_regionaxes_0x_difference',
                          'standard_sp_regionaxes_0y_sum',
                          'standard_sp_regionaxes_0y_difference',

                          'standard_sp_regionaxes_1x_sum',
                          'standard_sp_regionaxes_1x_difference',
                          'standard_sp_regionaxes_1y_sum',
                          'standard_sp_regionaxes_1y_difference' ]
        
        features_df = rag.compute_features(values, feature_names)
        assert len(features_df) == len(rag.edge_ids)
        assert list(features_df.columns.values) == output_columns, \
            "Wrong output feature names: {}".format( features_df.columns.values )

        # Using shorthand names should result in the same columns
        features_df = rag.compute_features(values, ['standard_sp_regionradii', 'standard_sp_regionaxes'])
        assert len(features_df) == len(rag.edge_ids)
        assert list(features_df.columns.values) == output_columns, \
            "Wrong output feature names: {}".format( features_df.columns.values )

        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # Check dtypes (pandas makes it too easy to get this wrong).
        dtypes = { colname: series.dtype for colname, series in features_df.iteritems() }
        assert all(dtype != np.float64 for dtype in dtypes.values()), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        # Manually compute the 'radius' of each superpixel
        # This is easy because each superpixel is just 1 column wide.
        col_coord_mean = np.arange(100).mean()
        centralized_col_coords = np.arange(100) - col_coord_mean
        col_coord_variance = centralized_col_coords.dot(centralized_col_coords)/100.
        col_radius = np.sqrt(col_coord_variance).astype(np.float32)

        for row_tuple in features_df.itertuples():
            row = OrderedDict( list(zip(['index'] + list(features_df.columns.values),
                                   row_tuple)) )
            # The superpixels were just vertical columns
            assert row['standard_sp_regionradii_0_sum'] == 2*col_radius
            assert row['standard_sp_regionradii_0_difference'] == 0.0
            assert row['standard_sp_regionradii_1_sum'] == 0.0
            assert row['standard_sp_regionradii_1_difference'] == 0.0

            # Axes are just parallel to the coordinate axes, so this is boring.
            # The x_sum is 0.0 because all superpixels are only 1 pixel wide in that direction.
            assert row['standard_sp_regionaxes_0x_sum'] == 0.0
            assert row['standard_sp_regionaxes_0x_difference'] == 0.0
            assert row['standard_sp_regionaxes_0y_sum'] == 2.0
            assert row['standard_sp_regionaxes_0y_difference'] == 0.0

            # The second axis is the smaller one, so here the y_sum axes are non-zero
            assert row['standard_sp_regionaxes_1x_sum'] == 2.0
            assert row['standard_sp_regionaxes_1x_difference'] == 0.0
            assert row['standard_sp_regionaxes_1y_sum'] == 0.0
            assert row['standard_sp_regionaxes_1y_difference'] == 0.0

    def test_edge_features_with_histogram(self):
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        feature_names = ['standard_edge_mean', 'standard_edge_minimum', 'standard_edge_maximum', 'standard_edge_variance',
                         'standard_edge_quantiles_25', 'standard_edge_quantiles_50', 'standard_edge_quantiles_75',
                         'standard_edge_count', 'standard_edge_sum']

        features_df = rag.compute_features(values, feature_names)
        assert len(features_df) == len(rag.edge_ids)
        assert list(features_df.columns.values) == ['sp1', 'sp2'] + list(feature_names), \
            "Wrong output feature names: {}".format( features_df.columns.values )

        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # Check dtypes (pandas makes it too easy to get this wrong).
        dtypes = { colname: series.dtype for colname, series in features_df.iteritems() }
        assert all(dtype != np.float64 for dtype in list(dtypes.values())), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        for row_tuple in features_df.itertuples():
            row = OrderedDict( list(zip(['index', 'sp1', 'sp2'] + list(feature_names),
                                   row_tuple)) )
            sp1 = row['sp1']
            sp2 = row['sp2']
            # Values were identical to the superpixels, so this is boring...
            assert row['standard_edge_mean'] == (sp1+sp2)/2.
            assert row['standard_edge_minimum'] == (sp1+sp2)/2.
            assert row['standard_edge_maximum'] == (sp1+sp2)/2.
            assert row['standard_edge_variance'] == 0.0
            assert row['standard_edge_quantiles_25'] == (sp1+sp2)/2.
            assert row['standard_edge_quantiles_75'] == (sp1+sp2)/2.
            assert row['standard_edge_count'] > 0
            assert row['standard_edge_sum'] == row['standard_edge_count'] * (sp1+sp2)/2.

    def test_edge_features_with_histogram_blank_data(self):
        """
        There was a bug related to histogram min/max if all edge pixels had the exact same value.
        In that case, min and max were identical and vigra complained.
        For example, if you give vigra histogramRange=(0.0, 0.0), that's a problem.
        This test verifies that no crashes occur in such cases.
        """
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        values = np.zeros_like(superpixels, dtype=np.float32)

        feature_names = ['standard_edge_mean', 'standard_edge_minimum', 'standard_edge_maximum', 'standard_edge_variance',
                         'standard_edge_quantiles_25', 'standard_edge_quantiles_50', 'standard_edge_quantiles_75',
                         'standard_edge_count', 'standard_edge_sum']

        features_df = rag.compute_features(values, feature_names)
        assert len(features_df) == len(rag.edge_ids)
        assert list(features_df.columns.values) == ['sp1', 'sp2'] + list(feature_names), \
            "Wrong output feature names: {}".format( features_df.columns.values )

        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # Check dtypes (pandas makes it too easy to get this wrong).
        dtypes = { colname: series.dtype for colname, series in features_df.iteritems() }
        assert all(dtype != np.float64 for dtype in list(dtypes.values())), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        for row_tuple in features_df.itertuples():
            row = OrderedDict( list(zip(['index', 'sp1', 'sp2'] + list(feature_names),
                                   row_tuple)) )

            assert row['standard_edge_mean'] == 0.0
            assert row['standard_edge_minimum'] == 0.0
            assert row['standard_edge_maximum'] == 0.0
            assert row['standard_edge_variance'] == 0.0
            assert row['standard_edge_quantiles_25'] == 0.0
            assert row['standard_edge_quantiles_75'] == 0.0
            assert row['standard_edge_count'] > 0
            assert row['standard_edge_sum'] == 0.0

    def test_edge_features_no_histogram(self):
        """
        Make sure vigra edge filters still work even if no histogram features are selected.
        """
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        feature_names = ['standard_edge_mean', 'standard_edge_minimum', 'standard_edge_maximum']

        features_df = rag.compute_features(values, feature_names)
        assert len(features_df) == len(rag.edge_ids)
        assert list(features_df.columns.values) == ['sp1', 'sp2'] + list(feature_names), \
            "Wrong output feature names: {}".format( features_df.columns.values )

        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        for row_tuple in features_df.itertuples():
            row = OrderedDict( list(zip(['index', 'sp1', 'sp2'] + list(feature_names),
                                   row_tuple)) )
            sp1 = row['sp1']
            sp2 = row['sp2']
            # Values were identical to the superpixels, so this is boring...
            assert row['standard_edge_mean'] == (sp1+sp2)/2.
            assert row['standard_edge_minimum'] == (sp1+sp2)/2.
            assert row['standard_edge_maximum'] == (sp1+sp2)/2.


    def test_shorthand_names(self):
        superpixels = generate_random_voronoi((100,200), 200)
        rag = Rag( superpixels )

        # For simplicity, just make values identical to superpixels
        values = superpixels.astype(np.float32)

        features_df = rag.compute_features(values, ['standard_edge_quantiles', 'standard_sp_quantiles'])
        
        quantile_names = ['quantiles_0', 'quantiles_10', 'quantiles_25', 'quantiles_50', 'quantiles_75', 'quantiles_90', 'quantiles_100']
        edge_feature_names = ['standard_edge_' + name for name in quantile_names]
        sp_features_names = ['standard_sp_' + name for name in quantile_names]
        
        sp_output_columns = []
        for name in sp_features_names:
            sp_output_columns.append( name + '_sum' )
            sp_output_columns.append( name + '_difference' )

        assert list(features_df.columns.values) == ['sp1', 'sp2'] + edge_feature_names + sp_output_columns

if __name__ == "__main__":
    import os
    import pytest
    module = os.path.split(__file__)[1][:-3]
    pytest.main(['-s', '--tb=native', '--pyargs', f'ilastikrag.tests.{module}'])
