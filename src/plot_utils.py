import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.interpolate import griddata

FIGSIZE = (9, 3)

class make_plot:
    '''
    class used to encapsulate plot operations,
    like showing or saving the plot
    '''
    def __init__(self, figsize=FIGSIZE, autoclose=True, save=None, subplots=False):
        self.figsize = figsize
        self.autoclose = autoclose
        self.save = save
        self.subplots = subplots

    def __enter__(self):
        if self.autoclose: plt.close('all')
        if self.subplots: return plt.subplots(figsize=self.figsize)
        return plt.figure(figsize=self.figsize)

    def __exit__(self, *_):
        plt.tight_layout()
        if self.save:
            import os
            os.makedirs(os.path.dirname(self.save), exist_ok=True)
            plt.savefig(str(self.save) if self.save.endswith('.png') else f'{self.save}.png')
        else: plt.show()
        if self.autoclose: plt.close()

def make_name(name, path=None):
    '''Create a valid name for save an image to file system'''
    if type(path) is str:
        if type(name) is dict:
            name = str.join('_', [
                str.join('__', [ str(vv) for vv in v ])
                if type(v) in (list, tuple) else str(v)
                for v in name.values()
            ]) 
        return f'{path}/{name}'
    return path

def plot_training_history(history, figsize=FIGSIZE, autoclose=True, save=None):
    '''Plots the training history of a model'''
    with make_plot(figsize, autoclose, save, True) as (_, ax):
        for metric in history.history.keys():
            ax.plot(history.history[metric], label=metric)
        if len(history.history.keys()) > 0: plt.legend()

def scatter(x, y, invert=False, figsize=FIGSIZE, autoclose=True, xlabel=None, ylabel=None, save=None, **kwargs):
    '''Plot a scatter chart alredy configured'''
    with make_plot(figsize, autoclose, save, True) as (_, ax):
        if type(y) is dict:
            for color, values in y.items():
                if invert: ax.scatter(values, x, c=color, **kwargs)
                else: ax.scatter(x, values, c=color, **kwargs)
        elif invert: ax.scatter(y, x, **kwargs)
        else: ax.scatter(x, y, **kwargs)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
    
def density(
    target,
    price=None,
    x=None,
    y=None,
    is_model=False,
    is_estimator=False,
    autoclose=True,
    discrete=False,
    color=cm.viridis,
    figsize=FIGSIZE,
    vmin=0.0,
    vmax=1.0,
    save=None
):
    '''Plot a density chart'''
    import tensorflow as tf
    import src.data_utils as ud


    if type(color) is str: color = getattr(cm, color)
    
    x, y = ud.grid_gen(x, y)
    
    if is_model:
        xy = (
            np.array([ month for unit in x for month in y ]),
            np.array([ unit for unit in x for month in y ])
        )
        z = target(xy)
        if hasattr(z, 'numpy'): z = z.numpy()
        z = z.squeeze().reshape((y.shape[0], x.shape[0])).T
    elif is_estimator:
        xy = dict(
            month=np.array([ month for unit in x for month in y ]),
            units=np.array([ unit for unit in x for month in y ])
        )
        fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x=xy,
            y=None,
            batch_size=1,
            num_epochs=1,
            shuffle=False
        )
        z = np.array(list(p.get('predictions')[0] for p in target.predict(input_fn=fn)))
    elif callable(target):
        z = np.array([
            [ target(month, unit) for unit in x ]
            for month in y
        ])
    else:
        target = target.copy()
        if price is not None: target['price'] = price
        target['month'], _, _ = ud.normalize(target['month'])
        target['units'], _, _ = ud.normalize(target['units'])
        target['price'], _, _ = ud.normalize(target['price'])
        xx, yy = (
            np.array([ month for unit in x for month in y ]),
            np.array([ unit for unit in x for month in y ])
        )
        z = griddata(
            target[['month', 'units']].values,
            target['price'].values,
            (xx, yy),
            method='linear'
        )
        z = z.reshape((y.shape[0], x.shape[0])).T
    
    z = z.reshape((y.shape[0], x.shape[0]))
    extent = [
        np.min(x), np.max(x),
        np.min(y), np.max(y)
    ]

    with make_plot(figsize, autoclose, save, subplots=True) as (fig, ax):
        if discrete: ax.imshow(z[::-1], cmap=color, vmin=vmin, vmax=vmax, extent=extent)
        else: ax.contourf(x, y, z, cmap=color, vmin=vmin, vmax=vmax, extent=extent)
        plt.ylabel('month')
        plt.xlabel('units')

def plot_pred_scatter(y_true, *y_pred, figsize=(10, 5), autoclose=True, save=None, **ys):
    '''
    Scatter plot that compares ground truth over predictions.
    The marked diagonal represents the ideal behavior.
    '''
    with make_plot(figsize, autoclose, save):
        for i, y in enumerate(list(y_pred) + list(ys.items())):
            if type(y) in (tuple, list): plt.scatter(y[1], y_true, marker='.', alpha=0.1, label=y[0])
            else: plt.scatter(y, y_true, marker='.', alpha=0.1, label=f'series_{i}')
    
        xl, xu = plt.xlim()
        yl, yu = plt.ylim()
        l, u = min(xl, yl), max(xu, yu)
        plt.plot([l, u], [l, u], ':', c='0.3')
        plt.xlim(l, u)
        plt.ylim(l, u)
        plt.xlabel('prediction')
        plt.ylabel('target')
        plt.legend()


