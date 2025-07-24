import pandas as pd
import numpy as np
from changeforest import Control, changeforest
def run_changeforest(data, feature_cols, minimal_relative_segment_length=0.05, minimal_gain_to_split=100,
                 gain_threshold=100):
    valid = data[data['shortNTPTime'].notna()].reset_index(drop=True)
    X = valid[feature_cols].apply(pd.to_numeric, errors='coerce').dropna().to_numpy()
    if len(X) < 2:
        return np.array([])

    # inintialize Changeforest control parameters
    control = Control(
        minimal_relative_segment_length=minimal_relative_segment_length,
        minimal_gain_to_split=minimal_gain_to_split,
        random_forest_n_estimators=100,
        random_forest_n_jobs=-1
    )
    result = changeforest(X, "random_forest", "bs", control)
    # extract change points from tree structure
    def extract_changepoints_with_gain(result):
        changepoints = []
        def traverse(node):
            if hasattr(node, 'best_split') and node.best_split is not None:
                changepoints.append({
                    'split': int(node.best_split),
                    'max_gain': node.max_gain,
                    'p_value': node.p_value
                })
                if hasattr(node, 'left') and node.left is not None:
                    traverse(node.left)
                if hasattr(node, 'right') and node.right is not None:
                    traverse(node.right)
        traverse(result)
        return pd.DataFrame(changepoints)

    changepoints_df = extract_changepoints_with_gain(result)
    # filter change points by gain threshold
    filtered = changepoints_df[changepoints_df['max_gain'] > gain_threshold]
    return np.sort(filtered['split'].values)