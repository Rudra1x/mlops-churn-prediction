import numpy as np
from scipy.stats import ks_2samp


def detect_drift(train_values, live_values, threshold=0.05):
    stat, p_value = ks_2samp(train_values, live_values)
    return p_value < threshold
