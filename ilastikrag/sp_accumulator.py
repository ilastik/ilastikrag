class SpAccumulator(object):
    def __init__(self, label_img, feature_names):
        pass
    
    def cleanup(self):
        raise NotImplementedError

    def ingest_values_for_block(self, label_block, value_block, block_start, block_stop):
        raise NotImplementedError
    
    def append_merged_sp_features_to_edge_df(self, edge_df):        
        raise NotImplementedError
