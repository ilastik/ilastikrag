import numpy as np
import pandas as pd

def append_vigra_features_to_dataframe( acc, df, feature_names):
    """
    Extract the specified features from the given RegionFeaturesAccumulator
    and append them as columns to the given DataFrame.

    Here we implement the logic for handling feature names that have 
    a suffix (e.g. 'quantiles_25').
    
    Parameters
    ----------
    feature_names
        High-level feature names with prefix and possible suffix, e.g. edge_vigra_quantiles_25

    acc
        A RegionFeatureAccumulator from which to extract the specified features.

    df
        A pandas.DataFrame to append the features to

    output_prefix
        Prefix column names with the given string.  Must be either 'edge_' or 'sp_'.
    """
    # Add a column for each feature we'll need
    vigra_feature_names = map(lambda name: name.split('_')[2], feature_names )
    for feature_name, vigra_feature_name in zip(feature_names, vigra_feature_names):
        if 'quantiles' in feature_name:
            quantile_suffix = feature_name.split('_')[-1]
            q_index = ['0', '10', '25', '50', '75', '90', '100'].index(quantile_suffix)
            df[feature_name] = pd.Series(acc['quantiles'][:, q_index], dtype=np.float32)
        elif 'regionradii' in feature_name:
            radii_suffix = feature_name.split('_')[-1]
            r_index = int(radii_suffix)
            df[feature_name] = pd.Series(acc['regionradii'][:, r_index], dtype=np.float32)
        elif 'regionaxes' in feature_name:
            suffix = feature_name.split('_')[-1]
            assert len(suffix) == 2
            r_index, axis = suffix
            axis_index = 'xyz'.index(axis) # vigra puts results in xyz order, regardless of array order.
            df[feature_name] = pd.Series(acc['regionaxes'][:, r_index, axis_index], dtype=np.float32)
        else:
            df[feature_name] = pd.Series(acc[vigra_feature_name], dtype=np.float32)
    
    return df

def get_vigra_feature_names(feature_names):
    """
    For the given list of feature names, return the list of feature names to compute in vigra.
    Basically, just remove prefixes and suffixes
    
    For example: ['edge_vigra_mean', 'sp_vigra_quantiles_25'] -> ['mean', 'quantiles']
    """
    feature_names = map(str.lower, feature_names)

    # drop  prefixes and quantile suffixes like '_25'
    vigra_feature_names = map(lambda name: name.split('_')[2], feature_names )
    
    # drop duplicates (from multiple quantile selections)
    return list(set(vigra_feature_names))

