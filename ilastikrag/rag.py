from collections import defaultdict, OrderedDict, namedtuple
from itertools import izip, imap, groupby

import numpy as np
import pandas as pd
import vigra

import logging
logger = logging.getLogger(__name__)

from .util import label_vol_mapping, edge_mask_for_axis, edge_ids_for_axis, \
                  unique_edge_labels, extract_edge_values_for_axis, nonzero_coord_array

from .accumulators.base import BaseEdgeAccumulator, BaseSpAccumulator
from .accumulators.standard import StandardEdgeAccumulator, StandardSpAccumulator, StandardFlatEdgeAccumulator
from .accumulators.similarity import SimilarityFlatEdgeAccumulator
from .accumulators.edgeregion import EdgeRegionEdgeAccumulator

class Rag(object):
    """
    Region Adjacency Graph
    
    Initialized with an ND label image of superpixels, and stores
    the edges between superpixels.
    
    +----------------------+------------------------------------------------------------------------------+
    | Attribute            | Description                                                                  |
    +======================+==============================================================================+
    | label_img            | The label volume you passed in.                                              |
    +----------------------+------------------------------------------------------------------------------+
    | sp_ids               | 1D ndarray of superpixel ID values, sorted.                                  |
    +----------------------+------------------------------------------------------------------------------+
    | max_sp               | The maximum superpixel ID in the label volume                                |
    +----------------------+------------------------------------------------------------------------------+
    | num_sp               | The number of superpixels in ``label_img``.                            |br|  |
    |                      | Not necessarily the same as ``max_sp``.                                |br|  |
    +----------------------+------------------------------------------------------------------------------+
    | num_edges            | The number of edges in the label volume.                                     |
    +----------------------+------------------------------------------------------------------------------+
    | edge_ids             | *ndarray, shape=(N,2)*                                                 |br|  |
    |                      | List of adjacent superpixel IDs, sorted. (No duplicates).              |br|  |
    |                      | *Guarantee:* For all edge_ids (sp1,sp2): sp1 < sp2.                    |br|  |
    +----------------------+------------------------------------------------------------------------------+
    | unique_edge_tables   | *dict* of *pandas.DataFrame* objects                                   |br|  |
    |                      | Columns: ``[sp1, sp2, edge_label]``, where ``edge_label``              |br|  |
    |                      | uniquely identifies each edge ``(sp1, sp2)`` within that table.        |br|  |
    |                      | See :py:attr:`unique_edge_tables` for details.                         |br|  |
    +----------------------+------------------------------------------------------------------------------+
    | dense_edge_tables    | *OrderedDict* of *pandas.DataFrame* objects (one per isotropic axis).  |br|  |
    |                      | Each DataFrame stores the id and location of all pixel                 |br|  |
    |                      | edge pairs in the volume *along a particular axis.*                    |br|  |
    |                      | See :py:attr:`dense_edge_tables` for details.                          |br|  |
    +----------------------+------------------------------------------------------------------------------+
    | flat_edge_label_img  | *ndarray, same shape as label_img except for the z-axis (1 px smaller)* |br| |
    |                      | If ``flat_superpixels=True``, this is a label volume for edges along    |br| |
    |                      | the z-axis, labeled according to the ``edge_label`` column from         |br| |
    |                      | :py:attr:`unique_edge_tables['z'] <unique_edge_tables>`.                |br| |
    +----------------------+------------------------------------------------------------------------------+

    **Limitations:**

    - This representation does not check for edge contiguity, so if two 
      superpixels are connected via multiple 'faces', those faces will both
      be lumped into one 'edge'.

    - No support for parallelization yet.
    """

    # Maintenance docs
    #
    """
    Implementation notes
    --------------------
    Internally, the edges along each axis are found independently and stored
    in separate pandas.DataFrame objects (one per axis in the volume).
    Every pixel face between two different superpixels is stored as a separate
    row in one of those DataFrames.
    
    This data structure's total RAM usage is proportional to the number of
    pixel faces on superpixel boundaries in the volume (i.e. the manhattan 
    distance of all superpixel boundaries interior to the label volume).
    It needs about 23 bytes per pixel face. (Each DataFrame row is 23 bytes.)
    
    Here are some example stats for a typical 512^3 cube of isotropic EM data:
    - 7534 superpixels
    - 53354 edges between superpixels
    - 19926582 (~20 million) individual edge pixel faces
    
    So, to handle that 0.5 GB label volume, this datastructure needs:
    20e6 pixel faces * 23 bytes == 0.46 GB of storage.
    
    Obviously, a volume with smaller superpixels will require more storage.
    
    TODO
    ----
    - Adding a function to merge two Rags should be trivial, if it seems useful
      (say, for parallelizing construction.)
    """

    # Used internally, during initialization
    _EdgeData = namedtuple("_EdgeData", "mask mask_coords ids forwardness")
    
    def __init__( self, label_img, flat_superpixels=False ):
        """
        Parameters
        ----------
        label_img
            *VigraArray*  |br|
            Label values do not need to be consecutive, but *excessively* high label values
            will require extra RAM when computing features, due to zeros stored
            within ``RegionFeatureAccumulators``.
        
        flat_superpixels
            *bool* |br|
            Set to ``True`` if ``label_img`` is a 3D volume whose superpixels are flat in the xy direction.
        """
        if isinstance(label_img, str) and label_img == '__will_deserialize__':
            return

        assert hasattr(label_img, 'axistags'), \
            "For optimal performance, make sure label_img is a VigraArray with accurate axistags"
        assert set(label_img.axistags.keys()).issubset('zyx'), \
            "Only axes z,y,x are permitted, not {}".format( label_img.axistags.keys() )
        assert label_img.dtype == np.uint32, \
            "label_img must have dtype uint32"
        assert not flat_superpixels or set('zyx').issubset(set(label_img.axistags.keys())), \
            "Can't use flat_superpixels with a 2D image."
        
        axes = 'zyx'[-label_img.ndim:]
        self._label_img = label_img.withAxes(axes)
        self._flat_superpixels = flat_superpixels
        
        edge_datas = OrderedDict()
        for axis, axiskey in enumerate(label_img.axistags.keys()):

            if flat_superpixels and axiskey == 'z':
                edge_mask = None # edge_ids_for_axis() supports edge_mask=None
                edge_mask_coords = None
            else:
                edge_mask = edge_mask_for_axis(label_img, axis)
                edge_mask_coords = nonzero_coord_array(edge_mask).transpose()
                
                # Save RAM: Convert to the smallest dtype we can get away with.
                if (np.array(label_img.shape) < 2**16).all():
                    edge_mask_coords = edge_mask_coords.astype(np.uint16)
                else:
                    edge_mask_coords = edge_mask_coords.astype(np.uint32)
                    
            edge_ids = edge_ids_for_axis(label_img, edge_mask, axis)
            edge_forwardness = edge_ids[:,0] < edge_ids[:,1]
            edge_ids.sort()

            edge_datas[axiskey] = Rag._EdgeData(edge_mask, edge_mask_coords, edge_ids, edge_forwardness)

        self._init_unique_edge_tables(edge_datas)
        self._init_dense_edge_tables(edge_datas)

        self._init_edge_ids()
        self._init_sp_attributes()
        if flat_superpixels:
            self._init_flat_edge_label_img(edge_datas)

    @property
    def label_img(self):
        return self._label_img

    @property
    def flat_superpixels(self):
        return self._flat_superpixels

    @property
    def sp_ids(self):
        return self._sp_ids

    @property
    def num_sp(self):
        return self._num_sp
    
    @property
    def max_sp(self):
        return self._max_sp

    @property
    def num_edges(self):
        all_axes = ''.join(self._label_img.axistags.keys())
        return len(self._unique_edge_tables[all_axes])

    @property
    def edge_ids(self):
        return self._edge_ids

    @property
    def flat_edge_label_img(self):
        if self._flat_superpixels:
            return self._flat_edge_label_img
        return None
    
    @property
    def unique_edge_tables(self):
        """
        *OrderedDict* of *pandas.DataFrame* objects.
        
        Each of these tables represents the set of edges that lie along a particular set of axes.

        If ``flat_superpixels=False``, then this dict contains just one item,
        with key ``zyx`` or ``yx``, depending on whether or not ``label_img`` is 3D or 2D.
        
        If ``flat_superpixels=True``, then this dict contains two tables for the disjoint
        sets of ``yx`` edges and ``z`` edges.  And additionally, it contains a third table in
        key ``zyx`` with all edges in the Rag (i.e. the superset of edges ``z``and ``yx``).

        Each table has columns: ``[sp1, sp2, edge_label]``, where ``edge_label`` 
        uniquely identifies each edge ``(sp1, sp2)`` within *that table*.

        .. note::
        
           Each table has an independent ``edge_label`` column. For a given edge
           ``(sp1,sp2)``, ``edge_label`` in table ``yx`` will not match the edge_label
           in table ``zyx``.
        """
        return self._unique_edge_tables

    @property
    def dense_edge_tables(self):
        """
        Read-only property.                                                    |br|
        A list of ``pandas.DataFrame`` objects (one per image axis).           |br|
        Each DataFrame stores the location and superpixel ids of all pixelwise |br|
        edge pairs in the volume *along a particular axis.*                    |br|

        **Example:**
        
        +---------+---------+-----------------+----------------+--------+--------+--------+
        | ``sp1`` | ``sp2`` | ``forwardness`` | ``edge_label`` | ``z``  | ``y``  | ``x``  |  
        +=========+=========+=================+================+========+========+========+
        |   1     |   2     |   True          |   10           |   0    |   10   |   13   |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        |   1     |   2     |   False         |   10           |   0    |   10   |   14   |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        |   1     |   2     |   False         |   10           |   0    |   10   |   15   |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        |   1     |   3     |   True          |   11           |   1    |   20   |   42   |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        |   1     |   3     |   True          |   11           |   1    |   20   |   43   |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        |   1     |   3     |   False         |   11           |   1    |   20   |   44   |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        | ...     | ...     | ...             | ...            | ...    | ...    | ...    |
        +---------+---------+-----------------+----------------+--------+--------+--------+
        
        **Column definitions:**
        
        +-----------------+----------------------------------------------------------------------------------------+
        | Column          | Description                                                                            |
        +=================+========================================================================================+
        | ``sp1``         | Superpixel ID                                                                          |
        +-----------------+----------------------------------------------------------------------------------------+
        | ``sp2``         | Superpixel ID. *Guarantee:* ``(sp1 < sp2)``                                            |
        +-----------------+----------------------------------------------------------------------------------------+
        | ``forwardness`` | ``True`` if ``sp1 < sp2``, otherwise ``False``.                                        |
        +-----------------+----------------------------------------------------------------------------------------+
        | ``edge_label``  | A ``uint32`` that uniquely identifies this ``(sp1,sp2)`` pair, regardless of axis.     |
        +-----------------+----------------------------------------------------------------------------------------+
        | ``z``           | Z-coordinate of this pixel edge                                                        |
        +-----------------+----------------------------------------------------------------------------------------+
        | ``y``           | Y-coordinate of this pixel edge                                                        |
        +-----------------+----------------------------------------------------------------------------------------+
        | ``x``           | X-coordinate of this pixel edge                                                        |
        +-----------------+----------------------------------------------------------------------------------------+
        
        """
        return self._dense_edge_tables

    def _init_unique_edge_tables(self, edge_datas):
        """
        Initialize the edge_label_lookup_df attribute.
        """
        all_axes = ''.join(self._label_img.axistags.keys())
        all_edge_ids = [t.ids for t in edge_datas.values()]

        self._unique_edge_tables = {}
        if not self._flat_superpixels:
            self._unique_edge_tables[all_axes] = unique_edge_labels( all_edge_ids )
        else:
            assert len(all_edge_ids) == 3
            assert edge_datas.keys() == list('zyx')
            unique_z = unique_edge_labels( [all_edge_ids[0]] )
            unique_yx = unique_edge_labels( all_edge_ids[1:] )
            unique_zyx = unique_edge_labels( [ unique_z[['sp1', 'sp2']].values,
                                               unique_yx[['sp1', 'sp2']].values ] )

            # If the superpixels are really flat, then unique_yx and unique_z
            # should be completely disjoint.
            assert len(unique_zyx) == ( len(unique_z) + len(unique_yx) )

            self._unique_edge_tables['z'] = unique_z
            self._unique_edge_tables['yx'] = unique_yx
            self._unique_edge_tables['zyx'] = unique_zyx

    def _init_edge_ids(self):
        # Tiny optimization:
        # Users will be accessing Rag.edge_ids over and over, so let's 
        # cache them now instead of extracting them on-the-fly
        all_axes = ''.join(self._label_img.axistags.keys())
        self._edge_ids = self._unique_edge_tables[all_axes][['sp1', 'sp2']].values

    def _init_flat_edge_label_img(self, edge_datas):
        assert self._flat_superpixels
        unique_table_z = self.unique_edge_tables['z']
        assert list(unique_table_z.columns.values) == ['sp1', 'sp2', 'edge_label']
        
        dense_table_z = pd.DataFrame(edge_datas['z'].ids, columns=['sp1', 'sp2'])
        dense_table_with_labels = pd.merge(dense_table_z, unique_table_z, on=['sp1', 'sp2'], how='left', copy=False)
        flat_edge_label_img = dense_table_with_labels['edge_label'].values
        
        shape = np.subtract(self._label_img.shape, (1, 0, 0))
        flat_edge_label_img.shape = tuple(shape)
        assert self._label_img.axistags.keys() == list('zyx')
        self._flat_edge_label_img = vigra.taggedView(flat_edge_label_img, 'zyx')

    def _init_dense_edge_tables(self, edge_datas):
        """
        Construct the N dense_edge_tables (one for each axis)
        """
        if self._flat_superpixels:
            dense_axes = 'yx'
        else:
            dense_axes = ''.join(self._label_img.axistags.keys())
        
        # Now create an dense_edge_table for each axis
        self._dense_edge_tables = OrderedDict()
        for axiskey in dense_axes:
            edge_data = edge_datas[axiskey]

            # Use uint32 index instead of deafult int64 to save ram            
            index_u32 = pd.Index(np.arange(len(edge_data.ids)), dtype=np.uint32)

            # Initialize with edge sp ids and directionality
            edge_table = pd.DataFrame( columns=['sp1', 'sp2', 'is_forward'],
                                       index=index_u32,
                                       data={ 'sp1': edge_data.ids[:, 0],
                                              'sp2': edge_data.ids[:, 1],
                                              'is_forward': edge_data.forwardness } )

            # Add 'edge_label' column. Note: pd.merge() is like a SQL 'join'
            dense_edge_table = pd.merge(edge_table, self._unique_edge_tables[dense_axes], on=['sp1', 'sp2'], how='left', copy=False)
            
            # Append columns for coordinates
            for key, coords, in zip(self._label_img.axistags.keys(), edge_data.mask_coords):
                dense_edge_table[key] = coords

            # Set column names
            coord_cols = self._label_img.axistags.keys()
            dense_edge_table.columns = ['sp1', 'sp2', 'forwardness', 'edge_label'] + coord_cols

            self._dense_edge_tables[axiskey] = dense_edge_table

    def _init_sp_attributes(self):
        """
        Compute and store our properties for sp_ids, num_sp, max_sp
        """
        all_axes = ''.join(self._label_img.axistags.keys())

        # Cache the unique sp ids to expose as an attribute
        # FIXME: vigra.unique() would be faster, and no implicit cast to int64
        unique_left = self._unique_edge_tables[all_axes]['sp1'].unique()
        unique_right = self._unique_edge_tables[all_axes]['sp2'].unique()
        self._sp_ids = pd.Series( np.concatenate((unique_left, unique_right)) ).unique()
        self._sp_ids = self._sp_ids.astype(np.uint32)
        self._sp_ids.sort()
        
        # We don't assume that SP ids are consecutive,
        # so num_sp is not the same as label_img.max()        
        self._num_sp = len(self._sp_ids)
        self._max_sp = self._sp_ids.max()


    # Initialize Rag.DEFAULT_ACCUMULATOR_CLASSES
    DEFAULT_ACCUMULATOR_CLASSES = {}
    for acc_cls in [StandardEdgeAccumulator, StandardSpAccumulator, StandardFlatEdgeAccumulator,
                    EdgeRegionEdgeAccumulator, SimilarityFlatEdgeAccumulator]:
        DEFAULT_ACCUMULATOR_CLASSES[(acc_cls.ACCUMULATOR_ID, acc_cls.ACCUMULATOR_TYPE)] = acc_cls

    def supported_features(self, accumulator_set="default"):
        """
        Return the set of available feature names to be used
        with this Rag and the given ``accumulator_set``.
        
        Parameters
        ----------
        accumulator_set:
            A list of acumulators to consider in addition to the built-in accumulators.
            If ``accumulator_set="default"``, then only the built-in accumulators are considered.
        
        Returns
        -------
        *list* of *str*
            The list acceptable feature names.
        """
        Rag._check_accumulator_conflicts(accumulator_set)

        feature_groups = {}
        if accumulator_set != "default":
            for acc in accumulator_set:
                feature_groups[(acc.ACCUMULATOR_ID, acc.ACCUMULATOR_TYPE)] = acc.supported_features()

        for (acc_id, acc_type) in Rag.DEFAULT_ACCUMULATOR_CLASSES.keys():
            if (acc_id, acc_type) not in feature_groups:
                acc_cls = Rag.DEFAULT_ACCUMULATOR_CLASSES[(acc_id, acc_type)]
                feature_groups[(acc_id, acc_type)] = acc_cls.supported_features(self)

        feature_names = []
        for group_names in feature_groups.values():
            feature_names += group_names
        return feature_names

    def compute_features(self, value_img, feature_names, edge_group=None, accumulator_set="default"):
        """
        The primary API function for computing features. |br|
        Returns a pandas DataFrame with columns ``['sp1', 'sp2', ...output feature names...]``

        Parameters
        ----------
        value_img
            *VigraArray*, same shape as ``self.label_img``.         |br|
            Pixel values are converted to ``float32`` internally.   |br|
            If your features are computed over the labels only,     |br|
            (not pixel values), you may pass ``value_img=None``     |br|
        
        feature_names
            *list of str*
            
            Feature names must have the following structure:
            
                ``<accumulator_id>_<type>_<feature>``.              |br|
            
            Example feature names:
                
                - ``standard_edge_count``
                - ``standard_edge_minimum``
                - ``standard_edge_variance``
                - ``standard_edge_quantiles_25``
                - ``standard_sp_count``
                - ``standard_sp_mean``

            The feature names are then passed to the appropriate ``EdgeAccumulator`` or ``SpAccumulator``.            
            See accumulator docs for details on supported feature names and their meanings.
            
            Features of type ``edge`` are computed only on the edge-adjacent pixels themselves.
            Features of type ``sp`` are computed over all values in the superpixels adjacent to
            an edge, and then converted into an edge feature, typically via sum or difference
            between the two superpixels.

        edge_group
            *str* or *list-of-str*                                                                |br|
            If ``Rag.flat_superpixels=True``, valid choices are ``'z'`` or ``'yx'``,
            or ``['z', 'yx']``, in which case an ``OrderedDict`` is returned with both results.
            
            For isotropic rags, there is only one valid choice, and it is selected by default:
            ``'zyx'`` (or ``'yx'`` if Rag is 2D).
        
        accumulator_set
            A list of acumulators to use in addition to the built-in accumulators.
            If ``accumulator_set="default"``, then only the built-in accumulators can be used.

        Returns
        -------
        *pandas.DataFrame*
            All unique superpixel edges in the volume,
            with computed features stored in the columns.

        Example
        -------
        ::

           >>> rag = Rag(superpixels)
           >>> feature_df = rag.compute_features(grayscale_img, ['standard_edge_mean', 'standard_sp_count'])
           >>> print list(feature_df.columns)
           ['sp1', 'sp2', 'standard_edge_mean', 'standard_sp_count_sum', 'standard_sp_count_difference']
        
        +---------+---------+------------------------+---------------------------+----------------------------------+
        | ``sp1`` | ``sp2`` | ``standard_edge_mean`` | ``standard_sp_count_sum`` | ``standard_sp_count_difference`` |
        +=========+=========+========================+===========================+==================================+
        | 1       | 2       | 123.45                 | 1000                      | 42                               |
        +---------+---------+------------------------+---------------------------+----------------------------------+
        | 1       | 3       | 234.56                 | 876                       | 83                               |
        +---------+---------+------------------------+---------------------------+----------------------------------+
        | ...     | ...     | ...                    | ...                       | ...                              |
        +---------+---------+------------------------+---------------------------+----------------------------------+

        """
        assert value_img is None or hasattr(value_img, 'axistags'), \
            "For optimal performance, make sure label_img is a VigraArray with accurate axistags"
        dense_axes =''.join(self.dense_edge_tables.keys())

        if self.flat_superpixels:
            valid_edge_groups = ('z', 'yx')
        else:
            valid_edge_groups = (''.join(self._label_img.axistags.keys()),)

        if edge_group is None:
            assert not self._flat_superpixels, "Must provide an edge_group"
            edge_group = dense_axes

        edge_group = str(edge_group)

        results = OrderedDict()
        if isinstance(edge_group, str):
            results[edge_group] = None
        else:
            for t in edge_group:
                results[t] = None
        assert all(edge_group in valid_edge_groups for edge_group in results.keys()), \
            "Unsupported edge_group."
        
        feature_groups = self._get_feature_groups(feature_names, accumulator_set)
        
        if dense_axes in results.keys():
            # Create a DataFrame for the results
            dense_axes = ''.join(self.dense_edge_tables.keys())
            dense_edge_ids = self.unique_edge_tables[dense_axes][['sp1', 'sp2']].values
            
            index_u32 = pd.Index(np.arange(len(dense_edge_ids)), dtype=np.uint32)
            edge_df = pd.DataFrame(dense_edge_ids, columns=['sp1', 'sp2'], index=index_u32)
    
            # Compute and append columns
            if 'edge' in feature_groups:
                edge_df = self._append_edge_features_for_values(edge_df, feature_groups['edge'], value_img, accumulator_set)
    
            if 'sp' in feature_groups:
                edge_df = self._append_sp_features_for_values(edge_df, feature_groups['sp'], value_img, accumulator_set)
            
            results[dense_axes] = edge_df

            # Typecheck the columns to help new accumulator authors spot problems in their code.
            dtypes = { colname: series.dtype for colname, series in edge_df.iterkv() }
            assert all(dtype != np.float64 for dtype in dtypes.values()), \
                "An accumulator returned float64 features. That's a waste of ram.\n"\
                "dtypes were: {}".format(dtypes)


        # FIXME: This recomputes the sp features
        if 'z' in results.keys():
            # Create a DataFrame for the results
            index_u32 = pd.Index(np.arange(len(self.unique_edge_tables['z'])), dtype=np.uint32)
            edge_df = pd.DataFrame(self.unique_edge_tables['z'][['sp1', 'sp2']].values, columns=['sp1', 'sp2'], index=index_u32)
    
            # Compute and append columns
            if 'flatedge' in feature_groups:
                edge_df = self._append_flatedge_features_for_values(edge_df, feature_groups['flatedge'], value_img, accumulator_set)
    
            if 'sp' in feature_groups:
                edge_df = self._append_sp_features_for_values(edge_df, feature_groups['sp'], value_img, accumulator_set)

            results['z'] = edge_df
            
            # Typecheck the columns to help new accumulator authors spot problems in their code.
            dtypes = { colname: series.dtype for colname, series in edge_df.iterkv() }
            assert all(dtype != np.float64 for dtype in dtypes.values()), \
                "An accumulator returned float64 features. That's a waste of ram.\n"\
                "dtypes were: {}".format(dtypes)

        if len(results) == 1:
            return results.values()[0]
        return results

    def _get_feature_groups(self, feature_names, accumulator_set="default"):
        """
        For the given list of feature_names, return features grouped in a dict:
            feature_groups[acc_type][acc_id] : [feature_name1, feature_name2, ...]
        """
        Rag._check_accumulator_conflicts(accumulator_set)

        feature_names = map(str.lower, feature_names)
        sorted_feature_names = sorted(feature_names, key=lambda name: name.split('_')[:2])

        # Group the names by type (edge/sp), then by accumulator ID,
        # but preserve the order of the features in each group (as a convenience to the user)
        feature_groups = defaultdict(dict)
        for (acc_id, acc_type), feature_group in groupby(sorted_feature_names,
                                                         key=lambda name: name.split('_')[:2]):
            feature_groups[acc_type][acc_id] = list(feature_group)

        # We only know about 'edge' and 'sp' features.
        unknown_feature_types = list(set(feature_groups.keys()) - set(['edge', 'sp', 'flatedge']))
        if unknown_feature_types:
            bad_names = feature_groups[unknown_feature_types[0]].values()[0]
            assert not unknown_feature_types, "Feature(s) have unknown type: {}".format(bad_names)

        return feature_groups

    def _append_edge_features_for_values(self, edge_df, edge_feature_groups, value_img, accumulator_set="default"):
        """
        Compute edge features and append them as columns to the given DataFrame.
        
        edge_df: DataFrame with columns (sp1, sp2) at least.
        edge_feature_groups: Dict of { accumulator_id : [feature_name, feature_name...] }
        value_img: ndarray of pixel values, or None
        accumulator_set: A list of additional accumulators to consider, or "default" to just use built-in.
        """
        # Extract values at the edge pixels
        if value_img is None:
            edge_values = None
        else:
            edge_values = OrderedDict()
            for axiskey, dense_edge_table in self.dense_edge_tables.items():
                axis_index = self._label_img.axistags.keys().index(axiskey)
                logger.debug("Axis {}: Extracting values...".format( axiskey ))
                coord_cols = self._label_img.axistags.keys()
                mask_coords = tuple(series.values for _colname, series in dense_edge_table[coord_cols].iteritems())
                edge_values[axiskey] = extract_edge_values_for_axis(axis_index, mask_coords, value_img)

        # Create an accumulator for each group
        for acc_id, feature_group_names in edge_feature_groups.items():
            edge_accumulator = self._select_accumulator_for_group(acc_id, 'edge', feature_group_names, accumulator_set)
            unsupported_names = set(feature_group_names) - set(edge_accumulator.supported_features(self))
            assert not unsupported_names, \
                "Some of your requested features aren't supported by this accumulator: {}".format(unsupported_names)
            
            with edge_accumulator:
                edge_accumulator.ingest_edges( self, edge_values )
                edge_df = edge_accumulator.append_edge_features_to_df(edge_df)

            # If the accumulator provided more features than the
            # user is asking for right now, remove the extra columns
            for colname in edge_df.columns.values[2:]:
                if '_edge_' in colname and not any(colname.startswith(name) for name in feature_group_names):
                    del edge_df[colname]

        return edge_df

    def _append_sp_features_for_values(self, edge_df, sp_feature_groups, value_img, accumulator_set="default"):
        """
        Compute superpixel-based features and append them as columns to the given DataFrame.
        
        edge_df: DataFrame with columns (sp1, sp2) at least.
        sp_feature_groups: Dict of { accumulator_id : [feature_name, feature_name...] }
        value_img: ndarray of pixel values, or None
        accumulator_set: A list of additional accumulators to consider, or "default" to just use built-in.
        """
        if isinstance(self._label_img, Rag._EmptyLabels):
            raise NotImplementedError("Can't compute superpixel-based features.\n"
                                      "You deserialized the Rag without deserializing the labels.")

        # Create an accumulator for each group
        for acc_id, feature_group_names in sp_feature_groups.items():
            sp_accumulator = self._select_accumulator_for_group(acc_id, 'sp', feature_group_names, accumulator_set)
            unsupported_names = set(feature_group_names) - set(sp_accumulator.supported_features(self))
            assert not unsupported_names, \
                "Some of your requested features aren't supported by this accumulator: {}".format(unsupported_names)

            with sp_accumulator:
                sp_accumulator.ingest_values(self, value_img)
                edge_df = sp_accumulator.append_edge_features_to_df(edge_df)

                # If the accumulator provided more features than the
                # user is asking for right now, remove the extra columns
                for colname in edge_df.columns.values[2:]:
                    if '_sp_' in colname and not any(colname.startswith(name) for name in feature_group_names):
                        del edge_df[colname]
        return edge_df

    def _append_flatedge_features_for_values(self, edge_df, flatedge_feature_groups, value_img, accumulator_set="default"):
        """
        Compute superpixel-based features and append them as columns to the given DataFrame.
        
        edge_df: DataFrame with columns (sp1, sp2) at least.
        flatedge_feature_groups: Dict of { accumulator_id : [feature_name, feature_name...] }
        value_img: ndarray of pixel values, or None
        accumulator_set: A list of additional accumulators to consider, or "default" to just use built-in.
        """
        if isinstance(self._label_img, Rag._EmptyLabels):
            raise NotImplementedError("Can't compute flatedge features.\n"
                                      "You deserialized the Rag without deserializing the labels.")

        # Create an accumulator for each group
        for acc_id, feature_group_names in flatedge_feature_groups.items():
            flatedge_accumulator = self._select_accumulator_for_group(acc_id, 'flatedge', feature_group_names, accumulator_set)
            unsupported_names = set(feature_group_names) - set(flatedge_accumulator.supported_features(self))
            assert not unsupported_names, \
                "Some of your requested features aren't supported by this accumulator: {}".format(unsupported_names)

            with flatedge_accumulator:
                flatedge_accumulator.ingest_values(self, value_img)
                edge_df = flatedge_accumulator.append_edge_features_to_df(edge_df)

                # If the accumulator provided more features than the
                # user is asking for right now, remove the extra columns
                for colname in edge_df.columns.values[2:]:
                    if '_flatedge_' in colname and not any(colname.startswith(name) for name in feature_group_names):
                        del edge_df[colname]
        return edge_df

    def edge_decisions_from_groundtruth(self, groundtruth_vol, asdict=False):
        """
        Given a reference segmentation, return a boolean array of "decisions"
        indicating whether each edge in this RAG should be ON or OFF for best
        consistency with the groundtruth.
        
        The result is returned in the same order as ``self.edge_ids``.
        An OFF edge means that the two superpixels are merged in the reference volume.
        
        If ``asdict=True``, return the result as a dict of ``{(sp1, sp2) : bool}``
        """
        assert (groundtruth_vol.shape == self._label_img.shape)
        sp_to_gt_mapping = label_vol_mapping(self._label_img, groundtruth_vol)

        unique_sp_edges = self.edge_ids
        decisions = sp_to_gt_mapping[unique_sp_edges[:, 0]] != sp_to_gt_mapping[unique_sp_edges[:, 1]]
    
        if asdict:
            return dict( izip(imap(tuple, unique_sp_edges), decisions) )
        return decisions

    def naive_segmentation_from_edge_decisions(self, edge_decisions, out=None ):
        """
        Given a list of ON/OFF labels for the Rag edges, compute a new label volume in which
        all supervoxels with at least one inactive edge between them are merged together.
        
        Requires ``networkx``.
        
        Parameters
        ----------
        edge_decisions
            1D bool array in the same order as ``self.edge_ids``                        |br|
            ``1`` means "active", i.e. the SP are separated across that edge, at least. |br|
            ``0`` means "inactive", i.e. the SP will be joined in the final result.     |br|
    
        out
            *VigraArray* (Optional).                                                    |br|
            Same shape as ``self.label_img``, but may have different ``dtype``.         |br|
        
        Returns
        -------
        *VigraArray*
        """
        import networkx as nx
        assert out is None or hasattr(out, 'axistags'), \
            "Must provide accurate axistags, otherwise performance suffers by 10x"
        assert edge_decisions.shape == (self._edge_ids.shape[0],)
    
        inactive_edge_ids = self.edge_ids[np.nonzero( np.logical_not(edge_decisions) )]
    
        logger.debug("Finding connected components in node graph...")
        g = nx.Graph( list(inactive_edge_ids) ) 
        
        # If any supervoxels are completely independent (not merged with any neighbors),
        # they haven't been added to the graph yet.
        # Add them now.
        g.add_nodes_from(self.sp_ids)
        
        sp_mapping = {}
        for i, sp_ids in enumerate(nx.connected_components(g), start=1):
            for sp_id in sp_ids:
                sp_mapping[int(sp_id)] = i
        del g
    
        return vigra.analysis.applyMapping( self._label_img, sp_mapping, out=out )

    def serialize_hdf5(self, h5py_group, store_labels=False, compression='lzf', compression_opts=None):
        """
        Serialize the Rag to the given hdf5 group.

        Parameters
        ----------
        h5py_group
            *h5py.Group*                                                       |br|
            Where to store the data. Should not hold any other data.
            
        store_labels
            If True, the labels will be stored as a (compressed) h5py Dataset. |br|
            If False, the labels are *not* stored, but you are responsible     |br|
            for loading them separately when calling _dataframe_to_hdf5(),     |br|
            unless you don't plan to use superpixel features.
        
        compression
            Passed directly to ``h5py.Group.create_dataset``.
        
        compression_opts
            Passed directly to ``h5py.Group.create_dataset``.
        """
        # Flag: flat_superpixels
        h5py_group.create_dataset('flat_superpixels', data=self.flat_superpixels)
        
        # Dense DFs
        dense_tables_parent_group = h5py_group.create_group('dense_edge_tables')
        for axiskey, df in self.dense_edge_tables.items():
            df_group = dense_tables_parent_group.create_group('{}'.format(axiskey))
            Rag._dataframe_to_hdf5(df_group, df)

        # Unique DFs
        unique_tables_parent_group = h5py_group.create_group('unique_edge_tables')
        for axiskey, df in self.unique_edge_tables.items():
            df_group = unique_tables_parent_group.create_group('{}'.format(axiskey))
            Rag._dataframe_to_hdf5(df_group, df)

        # label_img metadata
        labels_dset = h5py_group.create_dataset('label_img',
                                                shape=self._label_img.shape,
                                                dtype=self._label_img.dtype,
                                                compression=compression,
                                                compression_opts=compression_opts)
        labels_dset.attrs['axistags'] = self.label_img.axistags.toJSON()
        labels_dset.attrs['valid_data'] = False

        # label_img contents
        if store_labels:
            # Copy and compress.
            labels_dset[:] = self._label_img
            labels_dset.attrs['valid_data'] = True

        # Z edge-label image
        if self._flat_superpixels:
            flat_edge_labels_dset = h5py_group.create_dataset('flat_edge_labels',
                                                              shape=self._flat_edge_label_img.shape,
                                                              dtype=self._flat_edge_label_img.dtype,
                                                              compression=compression,
                                                              compression_opts=compression_opts,
                                                              data=self.flat_edge_label_img)
            flat_edge_labels_dset.attrs['axistags'] = self.flat_edge_label_img.axistags.toJSON()


    @classmethod
    def deserialize_hdf5(cls, h5py_group, label_img=None):
        """
        Deserialize the Rag from the given ``h5py.Group``,
        which was written via ``Rag.serialize_to_hdf5()``.

        Parameters
        ----------
        label_img
            If not ``None``, don't load labels from hdf5, use this volume instead.
            Useful for when ``serialize_hdf5()`` was called with ``store_labels=False``. 
        """
        rag = Rag('__will_deserialize__')

        # Flag: flat_superpixels
        rag._flat_superpixels = h5py_group['flat_superpixels'][()]
        
        # Dense Edge DFs
        rag._dense_edge_tables = OrderedDict()
        dense_tables_parent_group = h5py_group['dense_edge_tables']
        for axiskey, df_group in sorted(dense_tables_parent_group.items())[::-1]: # tables should be restored to zyx order.
            rag._dense_edge_tables[axiskey] = Rag._dataframe_from_hdf5(df_group)

        # Dense Edge DFs
        rag._unique_edge_tables = {}
        unique_tables_parent_group = h5py_group['unique_edge_tables']
        for axiskey, df_group in sorted(unique_tables_parent_group.items()):
            rag._unique_edge_tables[axiskey] = Rag._dataframe_from_hdf5(df_group)
        
        # label_img
        label_dset = h5py_group['label_img']
        axistags = vigra.AxisTags.fromJSON(label_dset.attrs['axistags'])
        if label_dset.attrs['valid_data']:
            assert not label_img, \
                "The labels were already stored to hdf5. Why are you also providing them externally?"
            label_img = label_dset[:]
            rag._label_img = vigra.taggedView( label_img, axistags )
        elif label_img is not None:
            assert hasattr(label_img, 'axistags'), \
                "For optimal performance, make sure label_img is a VigraArray with accurate axistags"
            assert set(label_img.axistags.keys()).issubset('zyx'), \
                "Only axes z,y,x are permitted, not {}".format( label_img.axistags.keys() )
            rag._label_img = label_img
        else:
            rag._label_img = Rag._EmptyLabels(label_dset.shape, label_dset.dtype, axistags)

        if rag._flat_superpixels:
            flat_edge_labels_dset = h5py_group['flat_edge_labels']
            flat_edge_labels = flat_edge_labels_dset[:]
            axistags = vigra.AxisTags.fromJSON(flat_edge_labels_dset.attrs['axistags'])
            rag._flat_edge_label_img = vigra.taggedView( flat_edge_labels, axistags )

        # Other attributes
        rag._init_edge_ids()
        rag._init_sp_attributes()

        return rag

    @classmethod
    def _dataframe_to_hdf5(cls, h5py_group, df):
        """
        Helper function to serialize a pandas.DataFrame to an h5py.Group.

        Note: This function uses a custom storage format,
              not the same format as pandas.DataFrame.to_hdf().

        Known to work for the DataFrames used in this file,
        including the MultiIndex columns in the dense_edge_tables.
        Not tested with more complicated DataFrame structures. 
        """
        h5py_group['row_index'] = df.index.values
        h5py_group['column_index'] = repr(df.columns.values)
        columns_group = h5py_group.create_group('columns')
        for col_index, col_name in enumerate(df.columns.values):
            columns_group['{:03}'.format(col_index)] = df[col_name].values

    @classmethod
    def _dataframe_from_hdf5(cls, h5py_group):
        """
        Helper function to deserialize a pandas.DataFrame from an h5py.Group,
        as written by Rag._dataframe_to_hdf5().

        Note: This function uses a custom storage format,
              not the same format as pandas.read_hdf().

        Known to work for the DataFrames used in this file,
        including the MultiIndex columns in the dense_edge_tables.
        Not tested with more complicated DataFrame structures. 
        """
        from numpy import array # We use eval() for the column index, which uses 'array'
        array # Avoid linter usage errors
        row_index_values = h5py_group['row_index'][:]
        column_index_names = list(eval(h5py_group['column_index'][()]))
        if isinstance(column_index_names[0], np.ndarray):
            column_index_names = map(tuple, column_index_names)
            column_index = pd.MultiIndex.from_tuples(column_index_names)
        elif isinstance(column_index_names[0], str):
            column_index = column_index_names
        else:
            raise NotImplementedError("I don't know how to handle that type of column index.: {}"
                                      .format(h5py_group['column_index'][()]))

        columns_group = h5py_group['columns']
        col_values = []
        for _name, col_values_dset in sorted(columns_group.items()):
            col_values.append( col_values_dset[:] )
        
        return pd.DataFrame( index=row_index_values,
                             columns=column_index,
                             data={ name: values for name,values in zip(column_index_names, col_values) } )

    class _EmptyLabels(object):
        """
        A little stand-in object for a missing labels array, in case the user
        wants to deserialize the Rag without a copy of the original labels.
        All functions in Rag can work with this object, except for
        SP computation, which needs the original label image.
        """
        def __init__(self, shape, dtype, axistags):
            object.__setattr__(self, 'shape', shape)
            object.__setattr__(self, 'dtype', dtype)
            object.__setattr__(self, 'axistags', axistags)
            object.__setattr__(self, 'ndim', len(shape))

        def _raise_NotImplemented(self, *args, **kwargs):
            raise NotImplementedError("Labels were not deserialized from hdf5.")
        
        # Accessing any function or attr other than those defined in __init__ will fail.
        __add__ = __radd__ = __mul__ = __rmul__ = __div__ = __rdiv__ = \
        __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = \
        __mod__ = __rmod__ = __pos__ = __neg__ = __call__ = \
        __getitem__ = __lt__ = __le__ = __gt__ = __ge__ = \
        __complex__ = __pow__ = __rpow__ = \
        __str__ = __repr__ = __int__ = __float__ = \
        __setattr__ = \
            _raise_NotImplemented
        
        def __getattr__(self, k):
            try:
                return object.__getattr__(self, k)
            except AttributeError:
                self._raise_NotImplemented()

    def _select_accumulator_for_group(self, acc_id, acc_type, feature_group_names, accumulator_set="default"):
        """
        Select an accumulator from the given accumulator_set for the given id/type and feature names.
        """
        if accumulator_set == "default":
            accumulator_set = []

        for acc in accumulator_set:
            assert acc.ACCUMULATOR_TYPE in ('edge', 'sp', 'flatedge'), \
                "{} has unknown accumulator-type: {}".format( acc, acc.ACCUMULATOR_TYPE )
            assert acc.ACCUMULATOR_ID, \
                "{} has empty accumulator-id: {}".format( acc, acc.ACCUMULATOR_ID )
            assert '_' not in acc.ACCUMULATOR_ID, \
                "{} has a bad char in its accumulator-id: {}".format( acc, acc.ACCUMULATOR_ID )

            if acc.ACCUMULATOR_ID == acc_id and acc.ACCUMULATOR_TYPE == acc_type:
                return acc

        # Try default
        return self._create_default_accumulator(acc_id, acc_type, feature_group_names)

    def _create_default_accumulator(self, acc_id, acc_type, feature_group_names):
        """
        Select the default accumulator class with the given id/type, and construct
        a new instance with the given feature names.
        """
        try:
            acc_class = Rag.DEFAULT_ACCUMULATOR_CLASSES[(acc_id, acc_type)]
        except KeyError:
            raise RuntimeError("No known accumulator class for features: {}".format( feature_group_names ))
        return acc_class(self, feature_group_names)

    @classmethod
    def _check_accumulator_conflicts(cls, accumulator_set):
        """
        Check the given accumulator set for possible conflicts,
        i.e. if two of them have matching types/ids, then we can't choose between them.
        """
        if accumulator_set == "default":
            return

        counts = defaultdict(lambda: 0)
        for acc in accumulator_set:
            assert isinstance(acc, BaseEdgeAccumulator) or isinstance(acc, BaseSpAccumulator), \
                "All accumulators must inherit from an accumulator base class.\n"\
                "Wrong type: {}".format( acc )
            
            counts[(acc.ACCUMULATOR_ID, acc.ACCUMULATOR_TYPE)] += 1
            if counts[(acc.ACCUMULATOR_ID, acc.ACCUMULATOR_TYPE)] > 1:
                raise RuntimeError("Conflicting accumulator selections.\n"
                                   "Multiple accumulators found to process features of type: {}_{}"
                                   .format(acc.ACCUMULATOR_ID, acc.ACCUMULATOR_TYPE))

if __name__ == '__main__':
    import sys
    logger.addHandler( logging.StreamHandler(sys.stdout) )
    logger.setLevel(logging.DEBUG)

    from lazyflow.utility import Timer
    
    import h5py
    #watershed_path = '/magnetic/data/flyem/chris-two-stage-ilps/volumes/subvol/256/watershed-256.h5'
    #grayscale_path = '/magnetic/data/flyem/chris-two-stage-ilps/volumes/subvol/256/grayscale-256.h5'

    watershed_path = '/magnetic/data/flyem/chris-two-stage-ilps/volumes/subvol/512/watershed-512.h5'
    grayscale_path = '/magnetic/data/flyem/chris-two-stage-ilps/volumes/subvol/512/grayscale-512.h5'
    
    logger.info("Loading watershed...")
    with h5py.File(watershed_path, 'r') as f:
        watershed = f['watershed'][:]
    if watershed.shape[-1] == 1:
        watershed = watershed[...,0]
    watershed = vigra.taggedView( watershed, 'zyx' )

    logger.info("Loading grayscale...")
    with h5py.File(grayscale_path, 'r') as f:
        grayscale = f['grayscale'][:]
    if grayscale.shape[-1] == 1:
        grayscale = grayscale[...,0]
    grayscale = vigra.taggedView( grayscale, 'zyx' )
    # typical features will be float32, not uint8, so let's not cheat
    grayscale = grayscale.astype(np.float32, copy=False)

    feature_names = []
    #feature_names = ['edgeregion_edge_regionradii', ]
    feature_names = ['standard_edge_mean', ]
    #feature_names += ['standard_edge_count', 'standard_edge_sum', 'standard_edge_mean', 'standard_edge_variance',
    #                  'standard_edge_minimum', 'standard_edge_maximum', 'standard_edge_quantiles_25', 'standard_edge_quantiles_50', 'standard_edge_quantiles_75', 'standard_edge_quantiles_100']
    #feature_names += ['standard_sp_count']
    #feature_names += ['standard_sp_count', 'standard_sp_sum', 'standard_sp_mean', 'standard_sp_variance', 'standard_sp_kurtosis', 'standard_sp_skewness']
    #feature_names += ['standard_sp_count', 'standard_sp_variance', 'standard_sp_quantiles_25', ]

    with Timer() as timer:
        logger.info("Creating python Rag...")
        rag = Rag( watershed )
    logger.info("Creating rag ({} superpixels, {} edges) took {} seconds"
                .format( rag.num_sp, rag.num_edges, timer.seconds() ))
    print "unique edge labels per axis: {}".format( [len(df['edge_label'].unique()) for df in rag.dense_edge_tables.values()] )
    print "Total pixel edges: {}".format( sum(len(df) for df in rag.dense_edge_tables ) )

    with Timer() as timer:
        edge_features_df = rag.compute_features(grayscale, feature_names)
        #edge_features_df = rag.compute_features(None, ['edgeregion_edge_regionradii'])
        
    print "Computing features with python Rag took: {}".format( timer.seconds() )
    #print edge_features_df[0:10]
    
    print ""
    print ""

#     # For comparison with vigra.graphs.vigra.graphs.regionAdjacencyGraph
#     import vigra
#     with Timer() as timer:
#         gridGraph = vigra.graphs.gridGraph(watershed.shape)
#         rag = vigra.graphs.regionAdjacencyGraph(gridGraph, watershed)
#         #ids = rag.uvIds()
#     print "Creating vigra Rag took: {}".format( timer.seconds() )
#  
#     from relabel_consecutive import relabel_consecutive
#     watershed = relabel_consecutive(watershed, out=watershed)
#     assert watershed.axistags is not None
#  
#     grayscale_f = grayscale.astype(np.float32, copy=False)
#     with Timer() as timer:
#         gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromImage(gridGraph,grayscale_f)
#         p0 = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)/255.0
#     print "Computing 1 vigra feature took: {}".format( timer.seconds() )
 

#     # For comparison with scikit-image Rag performance. (It's bad.)
#     from skimage.future.graph import RAG
#     with Timer() as timer:
#         logger.info("Creating skimage Rag...")
#         rag = RAG( watershed )
#     logger.info("Creating skimage rag took {} seconds".format( timer.seconds() ))
