class EdgeAccumulator(object):
    def __init__(self, label_img, feature_names):
        pass
    
    def cleanup(self):
        raise NotImplementedError

    def ingest_edges_for_block(self, axial_edge_dfs, block_start, block_stop):
        raise NotImplementedError
    
    def append_merged_edge_features_to_df(self, edge_df):
        raise NotImplementedError
    
