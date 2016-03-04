def append_vigra_features_to_dataframe( acc, df, feature_names, output_prefix):
    """
    Extract the specified features from the given RegionFeaturesAccumulator
    and append them as columns to the given DataFrame.

    Here we implement the logic for handling feature names that have 
    a suffix (e.g. 'quantiles_25').
    
    Parameters
    ----------
    feature_names
        High-level feature names that may or may not have a suffix, e.g. ``quantiles_25`` or ``mean``.

    acc
        A RegionFeatureAccumulator from which to extract the specified features.

    df
        A pandas.DataFrame to append the features to

    output_prefix
        Prefix column names with the given string.  Must be either 'edge_' or 'sp_'.
    """
    assert output_prefix in ('edge_', 'sp_')

    # Add a column for each feature we'll need
    for feature_name in feature_names:
        output_name = output_prefix + feature_name
        if feature_name.startswith('quantiles'):
            quantile_suffix = feature_name.split('_')[1]
            q_index = ['0', '10', '25', '50', '75', '90', '100'].index(quantile_suffix)
            df[output_name] = acc['quantiles'][:, q_index]
        else:
            df[output_name] = acc[feature_name]
    
    return df

def get_vigra_feature_names(feature_names):
    """
    For the given list of highlevel feature names, return the list of feature names to compute in vigra.
    Basically, just remove any suffixes like '_25'.
    
    For example: ['mean', 'quantiles_25'] -> ['mean', 'quantiles']
    """
    feature_names = map(str.lower, feature_names)

    # drop quantile suffixes like '_25'
    vigra_feature_names = map(lambda name: name.split('_')[0], feature_names )
    
    # drop duplicates (from multiple quantile selections)
    return list(set(vigra_feature_names))

