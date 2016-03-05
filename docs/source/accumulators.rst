.. currentmodule:: ilastikrag

.. _accumulators:

Feature Accumulators
--------------------

..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />

The features produced by :class:`Rag.compute_features()` are computed using "accumulator" classes.
There are two types of acccumulators: ``edge`` and ``sp``.  ``edge`` accumulators perform their
computation using only information about the edge pixels (and corresponding values) of every pair of 
adjacent superpixels, whereas ``sp`` features process the dense volume of labels and values.

All accumulators inherit from one of the two accumulator base classes: ``EdgeAccumulator`` and ``SpAccumulator``.

.. autoclass:: ilastikrag.EdgeAccumulatorBase

   .. autoattribute: ACCUMULATOR_TYPE
   .. autoattribute: ACCUMULATOR_ID
   .. automethod:: __init__   
   .. automethod:: cleanup   
   .. automethod:: ingest_edges_for_block   
   .. automethod:: append_merged_edge_features_to_df   

.. autoclass:: ilastikrag.SpAccumulatorBase

   .. autoattribute: ACCUMULATOR_TYPE
   .. autoattribute: ACCUMULATOR_ID
   .. automethod:: __init__   
   .. automethod:: cleanup   
   .. automethod:: ingest_values_for_block   
   .. automethod:: append_merged_sp_features_to_edge_df   


.. autoclass:: ilastikrag.accumulators.standard.StandardEdgeAccumulator

.. autoclass:: ilastikrag.accumulators.standard.StandardSpAccumulator
