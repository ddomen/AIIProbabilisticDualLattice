import numpy as np
import pandas as pd
from functools import reduce

def normalize(x):
    '''Normalize a vector: (x - min) / (MAX - min)'''
    m, M = x.min(), x.max()
    return (x - m) / (M - m), m, M

def denormalize(x, min, max):
    '''Denormalize a vector: x * (MAX - min) + min'''
    return x * (max - min) + min

def split_data(df, *splits, shuffle=False):
    """
    split data in to a given number of splits
    such that the proportions between them reflect
    the proportions between the given splits

    ----------------------------------------

    Parameters
    ----------

    df: numpy.ndarray | pandas.Dataframe
        the data to be splitted
    
    splits: List[int | float]
        the proportions between the splits 
    """
    length = len(df)
    total = np.sum(splits)
    splits = [ float(split) / total for split in splits ]
    splits = reduce(lambda acc, v: acc + [ acc[-1] + v ], splits, [0])
    return np.split(df.sample(frac=1) if shuffle else df, [ int(split * length) for split in splits[1:-1] ])

def grid_gen(x=None, y=None):
    '''Generate an equally spaced grid of the given shape.'''
    if x is None and y is None:
        x = (0, 1.0, 10)
        y = (0, 1.0, 10)
    elif x is None: x = y
    elif y is None: y = x

    if type(x) is int: x = np.linspace(0, 1.0, max(x, 3))
    elif type(x) in [list, tuple]: x = np.linspace(*x)
        
    if type(y) is int: y = np.linspace(0, 1.0, max(y, 3))
    elif type(y) in [list, tuple]: y = np.linspace(*y)
        
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] < 3: x = np.linspace(0, 1.0, 3)
    if y.shape[0] < 3: y = np.linspace(0, 1.0, 3)

    return x, y

def gen_dataset(fn, units=None,  months=12):
    '''
    Generate a dataset:
        - builds a grid interscting months and units
        - iterate for each cell of the grid evaluating the given function

    Returns a pd.DataFrame with the corresponding columns:

    | month | units |  price   |
    |:-----:|:-----:|:--------:|
    |   m   |   u   | fn(m, u) |
    '''
    x, y = grid_gen(x=units, y=months)
    return pd.DataFrame([
        [month, units, fn(month, units)]
        for month in y
        for units in x
    ], columns=['month', 'units', 'price'])


def preprocess(df, normalize=True):
    '''Common preprocess operations over month/units/price dataset'''
    df = df[
        (~df['date'].isnull()) &    # removing invalid dates
        (df['amount'] > 0) &        # removing invalid amount
        (df['units'] > 0) &         # removing invalid units
        (df['units'] < 1000)        # removing uint32 negative values (eg. 65535u == -1)
    ].copy()

    if normalize:
        def unpack(col): return col, col.min(), col.max()
    else:
        def unpack(col):
            m, M = col.min(), col.max()
            return (col - m) / (M - m), m, M
    
    # extracting year from date                         [ 2014 - 2020 ] -> [ 0 - 1 ]
    df['year']  , df['MIN_YEAR']  , df['MAX_YEAR']   = unpack(df['date'].dt.year)
    # extracting month from date                        [ 0 - 12 ] -> [ 0 - 1 ]
    df['month'] , df['MIN_MONTH'] , df['MAX_MONTH']  = unpack(df['date'].dt.month)
    # computing the average price of the transaction    [ 0 - 1 ] (the price is computed before normalization to prevent 0/0)
    df['price'] , df['MIN_PRICE'] , df['MAX_PRICE']  = unpack(df['amount'] / df['units'])
    # normalization of the amount                       [ 0 - 1 ]
    df['amount'], df['MIN_AMOUNT'], df['MAX_AMOUNT'] = unpack(df['amount'])
    # normalization of the units                        [ 0 - 1 ]
    df['units'] , df['MIN_UNITS'] , df['MAX_UNITS']  = unpack(df['units'])

    # ensuring both months and units are float for tensorflow compatibility
    df['month'] = df['month'].astype('float')
    df['units'] = df['units'].astype('float')

    return df

def aggregate(df, merge=True):
    '''Add the price mean and std for each row, aggregating by month and units'''
    df_agg = df.groupby(['month', 'units']).aggregate(dict(price=['std', 'mean'])).reset_index().fillna(0)
    df_agg.loc[:, 'mean'] = df_agg.loc[:, ('price', 'mean')]
    df_agg.loc[:, 'std'] = df_agg.loc[:, ('price', 'std')]
    del df_agg['price']
    
    if merge:
        df = df.merge(df_agg, on=['month', 'units']).rename(columns={ ('mean', ''): 'mean', ('std', ''): 'std' })

    return df, df_agg

def intra_class_norm(
    df,
    column='PRODUCT_ID',
    input='Amount',
    target='Price',
    output='AmountNorm'
):
    '''Normalize the input amount, dividing it by the mean of the given column (class).'''
    df.loc[:, '__SCALE'] = 1.0
    for _, group in df.groupby(column):
        df.loc[group.index, '__SCALE'] = group[target].mean()
    df.loc[:, output] = df[input] / df['__SCALE']
    del df['__SCALE']
    return df

def as_dataframe(df, extension='parquet', *args, **kwargs):
    '''
    Return the input dataframe or read it from the file system,
    with right extension and method (uses `pd.read_{extension}`)
    '''
    if type(df) is str: df = getattr(pd, f'read_{extension}')(df, *args, **kwargs)
    return df

def save_dataframe(df, filename, extension='parquet', *args, **kwargs):
    '''Saves a dataframe with the right extension and method (uses `pd._to{extension}`)'''
    getattr(df, f'to_{extension}')(filename, *args, **kwargs)
    return df

def add_metrics(file, *metrics, separator=';'):
    '''
    Add the given metrics to the csv file, ensuring
    consistency between headers and values
    '''
    import os
    import json
    dirname = os.path.realpath(os.path.dirname(file))
    os.makedirs(dirname, exist_ok=True)
    hasHeader = False
    if os.path.exists(file):
        with open(file, 'r') as f: rows = f.readlines()
        rows = [ [ r.strip() for r in row.split(separator) ] for row in rows ]
        if len(rows):
            header = rows[0]
            hasHeader = True
    if not hasHeader:
        header = [ 'name' ]
        rows = [ header ]
    for metric in metrics:
        for m in metric:
            if m.startswith('__'): continue
            if m not in header: header.append(m)
    for row in rows:
        for i in range(len(row)):
            if type(row[i]) is dict: row[i] = json.dumps(row[i])
        while len(row) < len(header): row.append('')
    for metric in metrics:
        rows.append([ str(metric.get(h, '')) for h in header ])
    with open(file, 'w') as f:
        f.write(str.join('\n', (str.join(separator, row) for row in rows)))