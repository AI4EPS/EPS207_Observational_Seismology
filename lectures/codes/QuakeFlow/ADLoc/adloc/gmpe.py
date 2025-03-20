import numpy as np


def calc_mag(amp, event_loc, station_loc, weight, min=-2, max=8):
    """
    Calculate magnitude from amplitude and distance
    Input:
        amp: amplitude in log10(cm/s)
        event_loc: event location (n, 3)
        station_loc: station location (n, 3)
        weight: weight for each observation (n,)
        min: minimum magnitude
        max: maximum magnitude
    Output:
        mag: magnitude
    """

    dist = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True)
    # mag_ = ( data - 2.48 + 2.76 * np.log10(dist) )
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    mag_ = (amp - c0 - c3 * np.log10(np.maximum(dist, 0.1))) / c1 + 3.5
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # mag_ = (data - c0 - c3*np.log10(dist))/c1
    # mag = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    # (Watanabe, 1971) https://www.jstage.jst.go.jp/article/zisin1948/24/3/24_3_189/_pdf/-char/ja
    # mag_ = 1.0/0.85 * (data + 1.73 * np.log10(np.maximum(dist, 0.1)) + 2.50)
    mu = np.sum(mag_ * weight) / (np.sum(weight) + 1e-6)
    std = np.sqrt(np.sum((mag_ - mu) ** 2 * weight) / (np.sum(weight) + 1e-12))
    mask = np.abs(mag_ - mu) <= 2 * std
    mag = np.sum(mag_[mask] * weight[mask]) / (np.sum(weight[mask]) + 1e-6)
    mag = np.clip(mag, min, max)
    return mag


def calc_amp(mag, event_loc, station_loc):
    """
    Calculate amplitude from magnitude and distance
    Input:
        mag: magnitude
        event_loc: event location (n, 3)
        station_loc: station location (n, 3)
    Output:
        logA: log amplitude in log10(cm/s)
    """
    dist = np.linalg.norm(event_loc - station_loc, axis=-1, keepdims=True)
    # logA = mag + 2.48 - 2.76 * np.log10(dist)
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    logA = c0 + c1 * (mag - 3.5) + c3 * np.log10(np.maximum(dist, 0.1))
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # logA = c0 + c1*mag + c3*np.log10(dist)
    # (Watanabe, 1971) https://www.jstage.jst.go.jp/article/zisin1948/24/3/24_3_189/_pdf/-char/ja
    # logA = 0.85 * mag - 2.50 - 1.73 * np.log10(np.maximum(dist, 0.1))
    return logA
