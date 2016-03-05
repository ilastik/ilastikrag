.. currentmodule:: ilastikrag.accumulators

.. _accumulators:

Feature Accumulators
====================

..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />

The features produced by :py:meth:`ilastikrag.Rag.compute_features()` are computed using "accumulator" classes.

There are two types of acccumulators: ``edge`` and ``sp``:

- ``edge`` accumulators process tables of **edge pixels** (and corresponding values)
- ``sp`` features process the **dense volume** of labels and values.

All accumulators inherit from one of the two accumulator base classes:

 - :class:`~ilastikrag.accumulators.BaseEdgeAccumulator`
 - :class:`~ilastikrag.accumulators.BaseSpAccumulator`

You can implement your own accumulators, but ``ilastikrag`` comes with the
following built-in accumulators, already activated by default:

- :class:`~ilastikrag.accumulators.standard.StandardEdgeAccumulator` 
- :class:`~ilastikrag.accumulators.standard.StandardSpAccumulator` 

.. _base_accumulators:

Base Accumulators
-----------------

.. autoclass:: BaseEdgeAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID
   .. automethod:: __init__   
   .. automethod:: cleanup   
   .. automethod:: ingest_edges_for_block   
   .. automethod:: append_merged_edge_features_to_df   

.. autoclass:: BaseSpAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID
   .. automethod:: __init__   
   .. automethod:: cleanup   
   .. automethod:: ingest_values_for_block   
   .. automethod:: append_merged_sp_features_to_edge_df   


.. currentmodule:: ilastikrag.accumulators.standard

.. _standard_accumulators:

Standard Accumulators
---------------------

.. autoclass:: ilastikrag.accumulators.standard.StandardEdgeAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID

   **Methods:** See :class:`~ilastikrag.BaseEdgeAccumulator`

.. autoclass:: ilastikrag.accumulators.standard.StandardSpAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID

   **Methods:** See :class:`~ilastikrag.BaseSpAccumulator`
