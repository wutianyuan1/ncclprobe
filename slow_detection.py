import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Rbeast as rb
from statsmodels.tsa import stattools
from scipy.signal import find_peaks
from scipy import stats


def remove_outliers(iter_durations, iter_start):
    iter_durations = pd.DataFrame({
        "data": iter_durations,
        "iter_start": iter_start
    })
    # Calculate the outlier boundary using z_score
    # Zscore = (data_point - mean) / std. deviation
    threshold_z = 2.0
    z = np.abs(stats.zscore(iter_durations['data']))
    outlier_indices = np.where(z > threshold_z)[0]
    iter_durations.drop(outlier_indices, inplace=True)
    return iter_durations['data'].to_numpy(), iter_durations['iter_start'].to_numpy()


def find_drop_naive(call_id, call_time, period, start):
    ## By dt, not work...
    df = pd.DataFrame({"call_id":call_id, "call_time":call_time})
    for i,d in df.groupby("call_id"):
        ts = d['call_time'].to_numpy()
        dt = ts[1:] - ts[:-1]
        plt.scatter(ts[:-1], dt, label=str(i))
    plt.tight_layout()
    plt.show()


def find_drop(call_id, call_time, period, start, thresh_prob=0.8, plot=False, plot_args=None):
    ## By period
    ts, iter_start = [], []
    for i in range(start, len(call_id), period):
        if i + period >= len(call_time):
            continue
        ts.append(call_time[i + period] - call_time[i])
        iter_start.append(call_time[i])
    ts, iter_start = remove_outliers(ts, iter_start)
    result = rb.beast(ts,season='none')
    rb.print(result)
    num_change_points = int(result.trend.ncp_mode[0])
    change_point_pos = np.array(result.trend.cp[:num_change_points], dtype=np.int32)
    change_point_prob = result.trend.cpPr[:num_change_points]
    ymax = max(ts)
    real_change_points = []
    for i in range(num_change_points):
        real_pos = iter_start[change_point_pos[i]]
        if change_point_prob[i] <= thresh_prob:
            continue
        print(f"Find change point at t={real_pos}, prob={change_point_prob[i]}")
        real_change_points.append(real_pos)
        if plot:
            ax = plot_args['ax']
            ax.plot([real_pos, real_pos], [0, ymax], c='black')
    if plot:
        ax = plot_args['ax']
        label = plot_args.get('label', 'Record')
        color = plot_args.get('color', 'blue')
        ax.scatter(iter_start, ts, label=label, color=color)
        ax.set_xlabel( plot_args.get('xlabel', 'X'))
        ax.set_ylabel( plot_args.get('ylabel', 'Y'))
    return real_change_points


def find_period(seq, nlags=30, significance_level=0.7):
    def dist(seq1, seq2):
        # Count the number of different elements in two sequence
        # We assume the NCCL call pattern is *exactly the same*
        # within each training iteration
        assert len(seq1) == len(seq2)
        return len(seq1) - np.sum(seq1 == seq2)
    acf_values = stattools.acf(seq, nlags=nlags)
    # Find peaks in the ACF that are above the significance level
    peaks, _ = find_peaks(acf_values, height=significance_level)
    if len(peaks) >= 1:
        # Estimate period as the lag of the first peak
        estimated_period = peaks[0]
        for i in range(len(seq)):
            if dist(seq[i:i + estimated_period],
                    seq[i + estimated_period: i + 2*estimated_period]) == 0:
                break
        print("Pattern starts from", i)
        return i, estimated_period
    else:
        warnings.warn("No peaks found in ACF, no patterns are found in NCCL logs")
        return None
