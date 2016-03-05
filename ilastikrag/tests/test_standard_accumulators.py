from collections import OrderedDict
import numpy as np

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
        dtypes = { colname: series.dtype for colname, series in features_df.iterkv() }
        assert all(dtype != np.float64 for dtype in dtypes.values()), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        # sp count features are normalized, consistent with the multicut paper.
        for _index, sp1, sp2, sp_count_sum, sp_count_difference in features_df.itertuples():
            assert sp_count_sum == np.power(sp_counts[sp1] + sp_counts[sp2], 1./superpixels.ndim).astype(np.float32)
            assert sp_count_difference == np.power(np.abs(sp_counts[sp1] - sp_counts[sp2]), 1./superpixels.ndim).astype(np.float32)

        # SUM
        features_df = rag.compute_features(values, ['standard_sp_sum'])
        assert len(features_df) == len(rag.edge_ids)
        assert (features_df.columns.values == ['sp1', 'sp2', 'standard_sp_sum_sum', 'standard_sp_sum_difference']).all()
        assert (features_df[['sp1', 'sp2']].values == rag.edge_ids).all()

        # sp sum features ought to be normalized, too...
        for _index, sp1, sp2, sp_sum_sum, sp_sum_difference in features_df.itertuples():
            assert sp_sum_sum == np.power(sp1*sp_counts[sp1] + sp2*sp_counts[sp2], 1./superpixels.ndim).astype(np.float32)
            assert sp_sum_difference == np.power(np.abs(sp1*sp_counts[sp1] - sp2*sp_counts[sp2]), 1./superpixels.ndim).astype(np.float32)

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
        dtypes = { colname: series.dtype for colname, series in features_df.iterkv() }
        assert all(dtype != np.float64 for dtype in dtypes.values()), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        # sp count features are normalized, consistent with the multicut paper.
        for _index, sp1, sp2, \
            sp_count_sum, sp_count_difference, \
            sp_quantiles_25_sum, sp_quantiles_25_difference, \
            sp_quantiles_75_sum, sp_quantiles_75_difference in features_df.itertuples():
            
            assert type(sp_quantiles_25_sum) == np.float32
            assert type(sp_quantiles_75_sum) == np.float32
            assert type(sp_quantiles_25_difference) == np.float32
            assert type(sp_quantiles_75_difference) == np.float32
            
            assert sp_count_sum == np.power(sp_counts[sp1] + sp_counts[sp2], 1./superpixels.ndim).astype(np.float32)
            assert sp_count_difference == np.power(np.abs(sp_counts[sp1] - sp_counts[sp2]), 1./superpixels.ndim).astype(np.float32)
            assert sp_quantiles_25_sum == float(sp1 + sp2), \
                "{} != {}".format( sp_quantiles_25_sum, float(sp1 + sp2) )
            assert sp_quantiles_75_sum == float(sp1 + sp2)
            assert sp_quantiles_25_difference == abs(float(sp1) - sp2)
            assert sp_quantiles_75_difference == abs(float(sp1) - sp2)

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
        dtypes = { colname: series.dtype for colname, series in features_df.iterkv() }
        assert all(dtype != np.float64 for dtype in dtypes.values()), \
            "An accumulator returned float64 features. That's a waste of ram.\n"\
            "dtypes were: {}".format(dtypes)

        for row_tuple in features_df.itertuples():
            row = OrderedDict( zip(['index', 'sp1', 'sp2'] + list(feature_names),
                                   row_tuple) )
            sp1 = row['sp1']
            sp2 = row['sp2']
            # Values were identical to the superpixels, so this is boring...
            assert np.isclose(row['standard_edge_mean'],  (sp1+sp2)/2.)
            assert np.isclose(row['standard_edge_minimum'], (sp1+sp2)/2.)
            assert np.isclose(row['standard_edge_maximum'], (sp1+sp2)/2.)
            assert np.isclose(row['standard_edge_variance'], 0.0)
            assert np.isclose(row['standard_edge_quantiles_25'], (sp1+sp2)/2.)
            assert np.isclose(row['standard_edge_quantiles_75'], (sp1+sp2)/2.)
            assert row['standard_edge_count'] > 0
            assert np.isclose(row['standard_edge_sum'], row['standard_edge_count'] * (sp1+sp2)/2.)

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
            row = OrderedDict( zip(['index', 'sp1', 'sp2'] + list(feature_names),
                                   row_tuple) )
            sp1 = row['sp1']
            sp2 = row['sp2']
            # Values were identical to the superpixels, so this is boring...
            assert np.isclose(row['standard_edge_mean'],  (sp1+sp2)/2.)
            assert np.isclose(row['standard_edge_minimum'], (sp1+sp2)/2.)
            assert np.isclose(row['standard_edge_maximum'], (sp1+sp2)/2.)

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
