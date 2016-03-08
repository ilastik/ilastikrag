.. currentmodule:: ilastikrag.rag

.. _rag:

Rag
---

..
   (The following |br| definition is the only way
   I can force numpydoc to display explicit newlines...) 

.. |br| raw:: html

   <br />

.. autoclass:: Rag

   .. automethod:: __init__
   .. automethod:: supported_features
   .. automethod:: compute_features
   .. automethod:: edge_decisions_from_groundtruth
   .. automethod:: naive_segmentation_from_edge_decisions
   .. automethod:: serialize_hdf5
   .. automethod:: deserialize_hdf5
   .. autoattribute:: axial_edge_dfs
   