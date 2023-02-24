"""
Implementation of the Deep Temporal Clustering model
Dataset loading functions

@author Florent Forest (FlorentF9)
@author Emmanuel Bresso
"""

from tslearn.utils import load_time_series_txt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def load_data(dataset_name):
    X = load_time_series_txt(dataset_name)
    y = None
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y