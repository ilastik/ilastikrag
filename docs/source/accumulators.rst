.. currentmodule:: ilastikrag.accumulators

.. _accumulators:

====================
Feature Accumulators
====================

..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />

The main work of :py:meth:`~ilastikrag.rag.Rag.compute_features()` is performed by "accumulator" classes.
There are two types of acccumulators: ``edge`` and ``sp``.

- ``edge`` accumulators process tables of **edge pixels** (and corresponding values)
- ``sp`` features process the **dense volume** of labels and values.

Built-in
========

- :class:`~ilastikrag.accumulators.standard.StandardEdgeAccumulator` 
- :class:`~ilastikrag.accumulators.standard.StandardSpAccumulator` 
- :class:`~ilastikrag.accumulators.edgeregion.EdgeRegionEdgeAccumulator` 


Base classes
============

You can write your own accumulator classes and pass them in an ``accumulator_set``
to :py:meth:`~ilastikrag.rag.Rag.compute_features()`.
All accumulators must inherit from one of these two base classes:

- :class:`~ilastikrag.accumulators.base.BaseEdgeAccumulator`
- :class:`~ilastikrag.accumulators.base.BaseSpAccumulator`


Reference
=========

.. _standard_accumulators:

Standard Accumulators
---------------------

.. currentmodule:: ilastikrag.accumulators.standard

.. autoclass:: ilastikrag.accumulators.standard.StandardEdgeAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID

   **Methods:** See :class:`~ilastikrag.accumulators.base.BaseEdgeAccumulator`

.. autoclass:: ilastikrag.accumulators.standard.StandardSpAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID

   **Methods:** See :class:`~ilastikrag.accumulators.base.BaseSpAccumulator`



.. _edgeregion_accumulator:

EdgeRegion Accumulator
----------------------

.. currentmodule:: ilastikrag.accumulators.edgeregion


.. autoclass:: ilastikrag.accumulators.edgeregion.EdgeRegionEdgeAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID

   **Methods:** See :class:`~ilastikrag.accumulators.base.BaseEdgeAccumulator`



.. _base_accumulators:

Base Accumulators
-----------------

.. currentmodule:: ilastikrag.accumulators.base

.. autoclass:: BaseEdgeAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID
   .. automethod:: __init__   
   .. automethod:: cleanup   
   .. automethod:: ingest_edges_for_block
   .. automethod:: append_merged_edge_features_to_df   
   .. automethod:: supported_features

.. autoclass:: BaseSpAccumulator

   .. autoattribute:: ACCUMULATOR_TYPE
   .. autoattribute:: ACCUMULATOR_ID
   .. automethod:: __init__   
   .. automethod:: cleanup   
   .. automethod:: ingest_values_for_block
   .. automethod:: append_merged_sp_features_to_edge_df   
   .. automethod:: supported_features

   