import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def get_column_names(features=[], indexes=[]):
    """
    Get the cartetian proudcts of features and indexes. For dropping columns in the dataframes.
    """
    column_names = []

    for index in indexes:
        for feature in features:
            column_names.append(f"{feature}[{index}]")
    
    return column_names

ACC_INDEXES = [1,2,3,4,5,6,8,9,10,11,13,14]
NEGPMAX_CLM_NAMES = get_column_names(['negpmax'], ACC_INDEXES)
NEGPMAX_XY_CLM_NAMES = NEGPMAX_CLM_NAMES + ['x', 'y']
PMAX_CLM_NAMES = get_column_names(['pmax'], ACC_INDEXES)
PMAX_XY_CLM_NAMES = PMAX_CLM_NAMES + ['x', 'y']



def mean_euclid_dist(y_true, y_pred):
    """
    Compute the mean euclidean distance between two series of 2D vectors.
    """
    diff = y_true - y_pred
    sqrd = diff**2
    summed = sqrd.sum(axis=1)
    euclid_dist = np.sqrt(summed)
    n = euclid_dist.shape[0]
    sigma = euclid_dist.sum(axis=0)

    mean_euclid_dist = sigma / n
    
    return mean_euclid_dist

def insert_zeros(data, acc_idxs, threshold, mode='training'):

    pd.set_option('mode.chained_assignment', None)

    df = data.copy()

    if mode == 'training':
        y = df[['x', 'y']].copy()

        X = df.drop(columns=['x', 'y'])

    elif mode == 'evaluation':
        X = df

        y = None

    df_pmax = df[[f'pmax[{i}]' for i in acc_idxs]]
    df_negpmax = df[[f'negpmax[{i}]' for i in acc_idxs]]
    df_area = df[[f'area[{i}]' for i in acc_idxs]]

    mask = df_pmax < threshold
    df_pmax[mask] = 0

    mask.columns = get_column_names(['negpmax'], acc_idxs)
    df_negpmax[mask] = 0

    mask.columns = get_column_names(['area'], acc_idxs)
    df_area[mask] = 0

    X = pd.concat([df_pmax, df_negpmax, df_area], axis=1)

    return X, y

def create_submission(y_pred):
    df = pd.DataFrame(data=y_pred, columns=['x', 'y'])
    ds_subm = df[['x', 'y']].astype(str).agg('|'.join, axis=1)
    df_subm = pd.concat([pd.Series(ds_subm.index.values), ds_subm], axis=1)
    df_subm.columns = ['Id', 'Predicted']

    return df_subm

def create_acc_map(y_valid, y_pred):
    diff = y_valid - y_pred
    sqrd = diff**2
    summed = sqrd.sum(axis=1)
    euclid_dist = np.sqrt(summed)
    euclid_dist

    arr = pd.concat([pd.DataFrame(y_valid), pd.DataFrame(euclid_dist)], axis=1)
    arr.columns=['x', 'y', 'ed']

    mli = arr.groupby(['x', 'y'])

    mean = mli.mean()

    mean_ed = mean.reset_index()

    plt.scatter(x=mean_ed['x'], y=mean_ed['y'], c=mean_ed['ed'], s=10)
    cbar = plt.colorbar(label='mean ed')
    plt.show()

def compute_negpmax_averages(df_dev: pd.DataFrame, threshold: int):
    df = df_dev.copy()
    df[df[NEGPMAX_CLM_NAMES] > 0] = -5

    # removing those with less than the threshold because they will impact average
    negpmax = df[NEGPMAX_CLM_NAMES]

    negpmax_below_thresh = negpmax[(negpmax < threshold).any(axis=1)]
    indexes = negpmax_below_thresh.index.values
    below_thresh_removed = df.drop(indexes)

    negpmax_below_thresh_removed = below_thresh_removed[NEGPMAX_XY_CLM_NAMES]

    xy_negpmax_average = negpmax_below_thresh_removed.groupby(by=['x', 'y']).mean().reset_index()

    return xy_negpmax_average, negpmax_below_thresh.index.values

def set_below_thresh_to_avg(df_dev, threshold: int, xy_negpmax_average, below_thresh_indexes):
    updated = df_dev.copy()

    negpmax_xy = df_dev[NEGPMAX_XY_CLM_NAMES]
    negpmax_xy_below_thresh = negpmax_xy.loc[below_thresh_indexes]

    clm_xy_index = dict()

    for clm in NEGPMAX_CLM_NAMES:
        series = negpmax_xy_below_thresh[clm]
        below = series[series < threshold]
        indexes = below.index.values
        xy = negpmax_xy_below_thresh.loc[indexes][['x', 'y']]
        if not xy.empty:
            clm_xy_index[clm] = xy

    for clm, df in clm_xy_index.items():
        indexes = df.index.values
        for index in indexes:
            x = df.loc[index]['x']
            y = df.loc[index]['y']
            avg_value = xy_negpmax_average[(xy_negpmax_average['x'] == x) & (xy_negpmax_average['y'] == y)][clm].values[0]
            updated[clm].loc[index] = avg_value

    return updated

def negpmax_maxvalues_below_10_to_avg(df_dev, xy_negpmax_average):
    negpmax = df_dev.copy()[NEGPMAX_CLM_NAMES]

    negpmax[negpmax > 0] = -15 # we need to do this to get the row 232393 in this part of the analysis
    max_values = negpmax.max(axis=1)

    max_below_minus_ten = max_values[max_values < -10]

    xy_below_minusten = df_dev.loc[max_below_minus_ten.index.values][['x', 'y']]

    averages = xy_below_minusten.reset_index().merge(xy_negpmax_average, how='left').set_index('index').drop(columns=['x', 'y'])

    df_dev.update(averages)

    return df_dev

def sqrd_above_thresh_to_avg(df_dev, threshold):
    df = df_dev.copy()

    pmax_clms = df[PMAX_CLM_NAMES]
    pmax_xy_clms = df[PMAX_XY_CLM_NAMES]

    xy_pmax_average = pmax_xy_clms.groupby(by=['x', 'y']).mean().reset_index()

    repeated = pd.DataFrame(np.repeat(xy_pmax_average.drop(columns=['x', 'y']).values, 100, axis=0))
    repeated.columns = PMAX_CLM_NAMES
    diff = pmax_clms - repeated
    sqrd = diff**2
    sum_of_squares = sqrd.sum(axis=1)

    ss_above_thresh = sum_of_squares[sum_of_squares > threshold]
    xy_above_thresh = df.loc[ss_above_thresh.index.values][['x', 'y']]
    averages = xy_above_thresh.reset_index().merge(xy_pmax_average, how='left').set_index('index')

    df.update(averages)

    return df