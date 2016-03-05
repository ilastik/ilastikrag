from collections import defaultdict
from itertools import izip, imap, groupby

import numpy as np
import pandas as pd
import vigra

import logging
logger = logging.getLogger(__name__)

from .util import label_vol_mapping, edge_mask_for_axis, edge_ids_for_axis, \
                  unique_edge_labels, extract_edge_values_for_axis, nonzero_coord_array

from .accumulators.vigra import VigraEdgeAccumulator
from .accumulators.vigra import VigraSpAccumulator

class Rag(object):
    """
    Region Adjacency Graph
    
    Initialized with an ND label image of superpixels, and stores
    the edges between superpixels.

    Attributes
    ----------

    label_img
        The label volume you passed in.

    sp_ids
        1D ndarray of superpixel ID values, sorted.

    max_sp
        The maximum superpixel ID in the label volume

    num_sp
        The number of superpixels in ``label_img``.                            |br|
        Not necessarily the same as ``max_sp``.                                |br|
    
    num_edges
        The number of edges in the label volume.

    edge_ids
        *ndarray, shape=(N,2)*                                                 |br|
        List of adjacent superpixel IDs, sorted.                               |br|
        *Guarantee:* For all edge_ids (u,v), u < v.                            |br|
        (No duplicates.)                                                       |br|
    
    edge_label_lookup_df
        *pandas.DataFrame*                                                     |br|
        Columns: ``[sp1, sp2, edge_label]``, where ``edge_label``              |br|
        uniquely identifies each edge ``(sp1, sp2)``.                          |br|

    axial_edge_dfs
        A list of ``pandas.DataFrame`` (one per axis).                         |br|
        Columns: ``['sp1', 'sp2', 'forwardness', 'edge_label', 'mask_coord']`` |br|
        Each DataFrame stores the id and location of all pixel                 |br|
        edge pairs in the volume *along a particular axis.*                    |br|
        See detailed description below.                                        |br|        
    """
    
    ##
    ## ADDITIONAL DEVELOPER DOCUMENTATION
    ##
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
    
    Limitations
    -----------
    - This representation does not check for edge contiguity, so if two 
      superpixels are connected via multiple 'faces', those faces will both
      be lumped into one 'edge'.

    - Coordinate-based features (e.g. RegionRadii) are not supported yet,
      for superpixels or edges.

    - No special treatment for anisotropic data yet.

    - No support for parallelization yet.
    
    TODO
    ----
    - Should SP features like 'mean' be weighted by SP size 
      before computing '_sum' and '_difference' columns for each edge?
    
    - Need to change API to allow custom feature functions.
    
    - Coordinate-based SP features would be easy to add (using vigra), but they aren't supported yet.
    
    - Coordinate-based edge features could be added without too much trouble, but not using vigra.
    
    - edge_count is computed 'manhattan' style, meaning that it
      is sensitive to the edge orientation (and so is edge_sum).
      Should we try to compensate for that somehow?
      Hmm... probably not. If we implement a RegionRadii edge feature,
      that's more informative than edge_count anyway, as long as it is
      implemented correctly (e.g. be sure to de-duplicate the edge coords
      after concatenating the edge points from each axis)
    
    - Basic support for anisotropic features will be easy, but perhaps not RAM efficient.
      Need to add 'axes' parameter to compute_highlevel_features().
    
    - Adding a function to merge two Rags should be trivial, if it seems useful
      (say, for parallelizing construction.)
    """

    # Clients can add their own accumulator classes by appending to this list.
    ACCUMULATOR_TYPES = [VigraEdgeAccumulator, VigraSpAccumulator]

    def __init__( self, label_img ):
        """
        Parameters
        ----------
        
        label_img
            *VigraArray*  |br|
            Label values do not need to be consecutive, but *excessively* high label values
            will require extra RAM when computing features, due to zeros stored
            within ``RegionFeatureAccumulators``.
        """
        # We do this here instead of above so users
        # can customize ACCUMULATOR_TYPES at runtime if desired.
        self._accumulator_class_lookup = { (acc_cls.ACCUMULATOR_TYPE, acc_cls.ACCUMULATOR_ID) : acc_cls
                                          for acc_cls in self.ACCUMULATOR_TYPES }

        if isinstance(label_img, str) and label_img == '__will_deserialize__':
            return

        assert hasattr(label_img, 'axistags'), \
            "For optimal performance, make sure label_img is a VigraArray with accurate axistags"
        assert set(label_img.axistags.keys()).issubset('zyx'), \
            "Only axes z,y,x are permitted, not {}".format( label_img.axistags.keys() )
        assert label_img.dtype == np.uint32, \
            "label_img must have dtype uint32"
        
        self._label_img = label_img
        
        edge_datas = []
        for axis in range(label_img.ndim):
            edge_mask = edge_mask_for_axis(label_img, axis)
            edge_ids = edge_ids_for_axis(label_img, edge_mask, axis)
            edge_forwardness = edge_ids[:,0] < edge_ids[:,1]
            edge_ids.sort()

            edge_mask_coords = nonzero_coord_array(edge_mask).transpose()
            
            # Save RAM: Convert to the smallest dtype we can get away with.
            if (np.array(label_img.shape) < 2**16).all():
                edge_mask_coords = edge_mask_coords.astype(np.uint16)
            else:
                edge_mask_coords = edge_mask_coords.astype(np.uint32)
                
            edge_datas.append( (edge_mask_coords, edge_ids, edge_forwardness) )

        self._init_final_edge_label_lookup_df(edge_datas)
        self._init_final_edge_ids()
        self._init_axial_edge_dfs(edge_datas)
        self._init_sp_attributes()

    @property
    def label_img(self):
        return self._label_img

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
        return len(self._final_edge_label_lookup_df)

    @property
    def edge_ids(self):
        return self._edge_ids

    @property
    def edge_label_lookup_df(self):
        return self._final_edge_label_lookup_df
    
    @property
    def axial_edge_dfs(self):
        """
        Read-only property.                                                    |br|
        A list of ``pandas.DataFrame`` objects (one per image axis).           |br|
        Each DataFrame stores the location and superpixel ids of all pixelwise |br|
        edge pairs in the volume *along a particular axis.*                    |br|

        These DataFrames are passed verbatim to ``EdgeAccumulator`` objects via their 
        ``ingest_edges_for_block()`` function, but with an extra column for the ``edge_value``.

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
        | ``mask_coord``  | N columns (e.g. ``z``, ``y``, ``x``) using a pandas multi-level index. |br|            |
        |                 | Stores coordinates of pixel just prior to (e.g. left of, above, etc.) each pixel edge. |
        +-----------------+----------------------------------------------------------------------------------------+
        
        """
        return self._axial_edge_dfs
    
    def _init_final_edge_label_lookup_df(self, edge_datas):
        """
        Initialize the edge_label_lookup_df attribute.
        """
        all_edge_ids = map(lambda t: t[1], edge_datas)
        self._final_edge_label_lookup_df = unique_edge_labels( all_edge_ids )

    def _init_final_edge_ids(self):
        """
        Initialize the edge_ids, and as a little optimization, RE-initialize the 
        final_edge_lookup, so its columns can be a view of the edge_ids
        """
        # Tiny optimization:
        # Users will be accessing edge_ids over and over, so let's extract them now
        self._edge_ids = self._final_edge_label_lookup_df[['sp1', 'sp2']].values

        # Now, to avoid having multiple copies of _edge_ids in RAM,
        # re-create final_edge_label_lookup_df using the cached edge_ids array
        index_u32 = pd.Index(np.arange(len(self._edge_ids)), dtype=np.uint32)
        self._final_edge_label_lookup_df = pd.DataFrame( index=index_u32,
                                                         data={'sp1': self._edge_ids[:,0],
                                                               'sp2': self._edge_ids[:,1],
                                                               'edge_label': self._final_edge_label_lookup_df['edge_label'].values } )

    def _init_axial_edge_dfs(self, edge_datas):
        """
        Construct the N axial_edge_df DataFrames (for each axis)
        """
        # Now create an axial_edge_df for each axis
        self._axial_edge_dfs = []
        for edge_data in edge_datas:
            edge_mask, edge_ids, edge_forwardness = edge_data

            # Use uint32 index instead of deafult int64 to save ram            
            index_u32 = pd.Index(np.arange(len(edge_ids)), dtype=np.uint32)

            # Initialize with edge sp ids and directionality
            axial_edge_df = pd.DataFrame( columns=['sp1', 'sp2', 'is_forward'],
                                          index=index_u32,
                                          data={ 'sp1': edge_ids[:, 0],
                                                 'sp2': edge_ids[:, 1],
                                                 'is_forward': edge_forwardness } )

            # Add 'edge_label' column. Note: pd.merge() is like a SQL 'join'
            axial_edge_df = pd.merge(axial_edge_df, self._final_edge_label_lookup_df, on=['sp1', 'sp2'], how='left', copy=False)
            
            # Append columns for coordinates
            for key, coords, in zip(self._label_img.axistags.keys(), edge_mask):
                axial_edge_df[key] = coords

            # For easier manipulation of the 'mask_coord' columns, set multi-level index for column names.
            combined_columns = [['sp1', 'sp2', 'forwardness', 'edge_label'] + len(self._label_img.axistags)*['mask_coord'],
                                [  '',     '',            '',           ''] + self._label_img.axistags.keys() ]
            axial_edge_df.columns = pd.MultiIndex.from_tuples(list(zip(*combined_columns)))

            self._axial_edge_dfs.append( axial_edge_df )

    def _init_sp_attributes(self):
        """
        Compute and store our properties for sp_ids, num_sp, max_sp
        """
        # Cache the unique sp ids to expose as an attribute
        unique_left = self._final_edge_label_lookup_df['sp1'].unique()
        unique_right = self._final_edge_label_lookup_df['sp2'].unique()
        self._sp_ids = pd.Series( np.concatenate((unique_left, unique_right)) ).unique()
        self._sp_ids.sort()
        
        # We don't assume that SP ids are consecutive,
        # so num_sp is not the same as label_img.max()        
        self._num_sp = len(self._sp_ids)
        self._max_sp = self._sp_ids.max()

    def compute_features(self, value_img, feature_names):
        """
        The primary API function for computing features. |br|
        Returns a pandas DataFrame with columns ``['sp1', 'sp2', ...output feature names...]``
        
        Parameters
        ----------
        value_img
            *VigraArray*, same shape as ``self.label_img``.         |br|
            Pixel values are converted to ``float32`` internally.
        
        feature_names:
            A list of feature names to compute.
            
            Feature names must have the following structure:
            
                ``<type>_<accumulator_id>_<feature>``.              |br|
            
            Example feature names:
                
                - ``edge_vigra_count``
                - ``edge_vigra_minimum``
                - ``edge_vigra_variance``
                - ``edge_vigra_quantiles_25``
                - ``sp_vigra_count``
                - ``sp_vigra_mean``

            The feature names are then passed to the appropriate ``EdgeAccumulator`` or ``SpAccumulator``.            
            See accumulator docs for details on supported feature names and their meanings.
            
            Features of type ``edge`` are computed only on the edge-adjacent pixels themselves.
            Features of type ``sp`` are computed over all values in the superpixels adjacent to
            an edge, and then converted into an edge feature, typically via sum or difference
            between the two superpixels.

        Returns
        -------
        A ``pandas.DataFrame`` for all unique superpixel edges in
        the volume, with computed features stored in the columns.

        Examples
        --------
        ::

           >>> feature_df = compute_features(grayscale_img,
           ...     ['edge_vigra_minimum', 'edge_vigra_maximum', 'sp_vigra_count', 'sp_vigra_mean'])
           >>>
           >>> feature_df.columns
           ['sp1', 'sp2', 'edge_vigra_minimum', 'edge_vigra_maximum', 'sp_vigra_count_sum', 'sp_vigra_count_difference', 'sp_vigra_mean_sum', 'sp_vigra_mean_difference']

        +-------------------------------+-----------------------------------------------------------------------------------+
        | Column                        | Description                                                                       |
        +===============================+===================================================================================+
        | ``sp1``                       | Superpixel ID.                                                                    |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``sp2``                       | Superpixel ID. *Guarantee:* ``(sp1 < sp2)``                                       |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``edge_vigra_minimum``        | Minimum value along the edge.                                                     |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``edge_vigra_maximum``        | Maximum value along the edge.                                                     |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``sp_vigra_count_sum``        | The sum of the sizes of the two superpixels adjacent to the edge.                 |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``sp_vigra_count_difference`` | The difference of the sizes of the two superpixels adjacent to the edge.          |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``sp_vigra_mean_sum``         | The sum of the mean intensity of the two superpixels adjacent to the edge.        |
        +-------------------------------+-----------------------------------------------------------------------------------+
        | ``sp_vigra_mean_difference``  | The difference of the mean intensity of the two superpixels adjacent to the edge. |
        +-------------------------------+-----------------------------------------------------------------------------------+        

        """
        feature_names = map(str.lower, feature_names)
        invalid_names = filter( lambda name: not( name.startswith('sp_') or name.startswith('edge_') ),
                                feature_names )
        assert not invalid_names, \
            "All feature names must start with either 'edge_' or 'sp_'. "\
            "Invalid names are: {}".format( feature_names )

        # Group the names by type (edge/sp), then by accumulator ID,
        # but preserve the order of the features in each group (as a convenience to the user)
        sorted_feature_names = sorted(feature_names, key=lambda name: name.split('_')[:2])
        feature_groups = defaultdict(dict)
        for (acc_type, acc_id), feature_group in groupby(sorted_feature_names,
                                                         key=lambda name: name.split('_')[:2]):
            feature_groups[acc_type][acc_id] = list(feature_group)

        # Create a DataFrame for the results
        index_u32 = pd.Index(np.arange(self.num_edges), dtype=np.uint32)
        edge_df = pd.DataFrame(self.edge_ids, columns=['sp1', 'sp2'], index=index_u32)

        # Compute and append columns
        if 'edge' in feature_groups:
            edge_df = self._append_edge_features_for_values(edge_df, feature_groups['edge'], value_img)

            dtypes = { colname: series.dtype for colname, series in edge_df.iterkv() }
            assert all(dtype != np.float64 for dtype in dtypes.values()), \
                "An accumulator returned float64 features. That's a waste of ram.\n"\
                "dtypes were: {}".format(dtypes)

        if 'sp' in feature_groups:
            edge_df = self._append_sp_features_for_values(edge_df, feature_groups['sp'], value_img)

            dtypes = { colname: series.dtype for colname, series in edge_df.iterkv() }
            assert all(dtype != np.float64 for dtype in dtypes.values()), \
                "An accumulator returned float64 features. That's a waste of ram.\n"\
                "dtypes were: {}".format(dtypes)

        return edge_df

    def _append_edge_features_for_values(self, edge_df, edge_feature_groups, value_img):
        """
        Compute edge features and append them as columns to the given DataFrame.
        
        edge_df: DataFrame with columns (sp1, sp2) at least.
        edge_feature_groups: Dict of { accumulator_id : [feature_name, feature_name...] }
        value_img: ndarray of pixel values
        """
        # For now, everything is with one big block
        block_start = value_img.ndim*(0,)
        block_stop = value_img.shape

        try:
            # Extract values at the edge pixels
            for axis, axial_edge_df in enumerate(self.axial_edge_dfs):
                logger.debug("Axis {}: Extracting values...".format( axis ))
                mask_coords = tuple(series.values for _colname, series in axial_edge_df['mask_coord'].iteritems())
                axial_edge_df['edge_value'] = extract_edge_values_for_axis(axis, mask_coords, value_img, aspandas=True)

            # Create an accumulator for each group
            for acc_id, feature_group_names in edge_feature_groups.items():
                try:
                    acc_class = self._accumulator_class_lookup[('edge', acc_id)]
                except KeyError:
                    raise RuntimeError("No known accumulator class for features: {}".format( feature_group_names ))

                with acc_class(self._label_img, feature_group_names) as edge_accumulator:
                    edge_accumulator.ingest_edges_for_block( self.axial_edge_dfs, block_start, block_stop )
                    edge_df = edge_accumulator.append_merged_edge_features_to_df(edge_df)
        finally:
            # Cleanup: Drop value columns
            for axial_edge_df in self.axial_edge_dfs:
                if 'edge_value' in axial_edge_df:
                    del axial_edge_df['edge_value']

        return edge_df

    def _append_sp_features_for_values(self, edge_df, sp_feature_groups, value_img):
        """
        Compute superpixel-based features and append them as columns to the given DataFrame.
        
        edge_df: DataFrame with columns (sp1, sp2) at least.
        sp_feature_groups: Dict of { accumulator_id : [feature_name, feature_name...] }
        value_img: ndarray of pixel values
        """
        # For now, everything is with one big block
        block_start = value_img.ndim*(0,)
        block_stop = value_img.shape

        # Create an accumulator for each group
        for acc_id, feature_group_names in sp_feature_groups.items():
            try:
                acc_class = self._accumulator_class_lookup[('sp', acc_id)]
            except KeyError:
                raise RuntimeError("No known accumulator class for features: {}".format( feature_group_names ))

            with acc_class(self._label_img, feature_group_names) as sp_accumulator:
                sp_accumulator.ingest_values_for_block(self._label_img, value_img, block_start, block_stop)
                edge_df = sp_accumulator.append_merged_sp_features_to_edge_df(edge_df)

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
        # Edge DFs
        axial_df_parent_group = h5py_group.create_group('axial_edge_dfs')
        for axis, axial_edge_df in enumerate(self.axial_edge_dfs):
            df_group = axial_df_parent_group.create_group('{}'.format(axis))
            Rag._dataframe_to_hdf5(df_group, axial_edge_df)

        # Final lookup DF
        lookup_df_group = h5py_group.create_group('final_edge_label_lookup_df')
        Rag._dataframe_to_hdf5(lookup_df_group, self._final_edge_label_lookup_df)

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
        
        # Edge DFs
        rag._axial_edge_dfs =[]
        axial_df_parent_group = h5py_group['axial_edge_dfs']
        for _name, df_group in sorted(axial_df_parent_group.items()):
            rag._axial_edge_dfs.append( Rag._dataframe_from_hdf5(df_group) )

        # Final lookup DF
        rag._final_edge_label_lookup_df = Rag._dataframe_from_hdf5( h5py_group['final_edge_label_lookup_df'] )
        
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

        # Other attributes
        rag._init_final_edge_ids()
        rag._init_sp_attributes()

        return rag

    @classmethod
    def _dataframe_to_hdf5(cls, h5py_group, df):
        """
        Helper function to serialize a pandas.DataFrame to an h5py.Group.

        Note: This function uses a custom storage format,
              not the same format as pandas.DataFrame.to_hdf().

        Known to work for the DataFrames used in this file,
        including the MultiIndex columns in the axial_edge_dfs.
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
        including the MultiIndex columns in the axial_edge_dfs.
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
    feature_names = ['edge_vigra_mean']
    #feature_names += ['edge_vigra_count', 'edge_vigra_sum', 'edge_vigra_mean', 'edge_vigra_variance',
    #                  'edge_vigra_minimum', 'edge_vigra_maximum', 'edge_vigra_quantiles_25', 'edge_vigra_quantiles_50', 'edge_vigra_quantiles_75', 'edge_vigra_quantiles_100']
    #feature_names += ['sp_vigra_count']
    #feature_names += ['sp_vigra_count', 'sp_vigra_sum', 'sp_vigra_mean', 'sp_vigra_variance', 'sp_vigra_kurtosis', 'sp_vigra_skewness']
    #feature_names += ['sp_vigra_count', 'sp_vigra_variance', 'sp_vigra_quantiles_25', ]

    with Timer() as timer:
        logger.info("Creating python Rag...")
        rag = Rag( watershed )
    logger.info("Creating rag ({} superpixels, {} edges) took {} seconds"
                .format( rag.num_sp, rag.num_edges, timer.seconds() ))
    print "unique edge labels per axis: {}".format( [len(df['edge_label'].unique()) for df in rag.axial_edge_dfs] )
    print "Total pixel edges: {}".format( sum(len(df) for df in rag.axial_edge_dfs ) )

    with Timer() as timer:
        #edge_features_df = rag.compute_highlevel_features(grayscale, feature_names)
        edge_features_df = rag.compute_features(grayscale, feature_names)
        
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
