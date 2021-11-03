'''
Probably the following error was fixed by a
newer version of tensorflow_probability;
the code below seems to be unnecessary.
Left for compatability reasons.
Some parts of this module were made
with lazy principles because of this.
'''
'''
copmat.v2 required for tensorflow probability
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
'''
import tensorflow as tf
import numpy as np

import src.metrics as um
import src.plot_utils as up

class NanLossCallback(tf.keras.callbacks.Callback):
    '''Stops the training if the loss become nan'''
    def __init__(self, verbose=False): self.verbose = verbose
    def on_train_batch_end(self, epoch, logs=None):
        if self.verbose and epoch and not (epoch % 1000): print(f'Epoch: {epoch}')
        if np.isnan(logs.get('loss')):
            self.model.stop_training = True
            print('\n=== NaN loss stopping ===')

def NNL(y, real):
    '''Negative log likelihood'''
    return real.log_prob(y)

def __make_distribution(distribution):
    '''
    Creates a distribution callback,
    used to creates DistributionLambda layers
    '''
    def distribution_lambda(tensor):
        return distribution(
            loc=tensor[..., :1],
            scale=tensor[..., 1:]
        )
    return distribution_lambda

DISTRIBUTIONS = None
def __get_distributions():
    '''Lazily retrieve a dictionary of all possible distributions'''
    global DISTRIBUTIONS
    if DISTRIBUTIONS is None:
        import tensorflow_probability as tfp
        DISTRIBUTIONS = dict(
            normal = __make_distribution(tfp.distributions.Normal),
            fixed_normal= lambda tensor: tfp.distributions.Normal(loc=tensor, scale=1e-1),
            laplace = __make_distribution(tfp.distributions.Laplace),
            cauchy = __make_distribution(tfp.distributions.Cauchy),
            gumbel = __make_distribution(tfp.distributions.Gumbel),
            poisson = __make_distribution(tfp.distributions.Poisson),
            exponential = __make_distribution(tfp.distributions.Exponential)
        )
    return DISTRIBUTIONS

def MLPRegressor(
    hidden=[42, 22, 16, 16, 10],
    hidden_activation='linear',
    activation='linear',
    inputs=2
):
    '''
    Simple fully-connected network.
    The head always produce Bx1 dimension output. (B = batch size)
    '''
    if type(inputs) is int: inputs = [1 for _ in range(inputs)]
    if type(hidden_activation) in (str, None):
        hidden_activation = [ hidden_activation ] * len(hidden)

    inputs = [ tf.keras.layers.Input(i) for i in inputs]
    curr = tf.keras.layers.Concatenate()(inputs)
    for h, a in zip(hidden, hidden_activation):
        curr = tf.keras.layers.Dense(h, activation=a)(curr)
    
    output = tf.keras.layers.Dense(1, activation=activation)(curr)
    return tf.keras.models.Model(inputs, [ output ])


def ProbabilisticRegressor(
    hidden=[32, 16],
    distribution='normal',
    hidden_activation='relu',
    activation='linear',
    inputs=2
):
    '''
    Fully-connected network which predicts mean and std,
    passed then to a Distribution Lambda Layer to sample the output
    '''
    import tensorflow_probability as tfp
    DST = __get_distributions()
    if type(inputs) is int: inputs = [1 for _ in range(inputs)]
    if type(hidden_activation) in (str, None):
        hidden_activation = [ hidden_activation ] * len(hidden)

    inputs = [ tf.keras.layers.Input(i) for i in inputs]
    curr = tf.keras.layers.Concatenate()(inputs)
    for h, a in zip(hidden, hidden_activation):
        curr = tf.keras.layers.Dense(h, activation=a)(curr)
    distribution = DST.get(distribution, DST['normal'])
    prob = tf.keras.layers.Dense(2, activation=activation)(curr)
    prob = tfp.layers.DistributionLambda(distribution)(prob)
    return tf.keras.models.Model(inputs, [ prob ])


def LatticeModel(sizes=[12, 24], mode=None):
    '''
    Creates a lattice model, with many inputs as sizes length.
    The mode argument can be:
    - `None`: no calibration, simplest model
    - `calib`: adds a PWLCalibration layer to the lattice
    - `calib-constr`: adds PWLCalibration with decraesing monotonicity,
        convex constraint, wrinkle kernel regularizer
    '''
    import tensorflow_lattice as tfl
    inputs = [ tf.keras.layers.Input((1, )) for _ in sizes ]
    
    if not (type(mode) is str): mode = ''
    if mode.startswith('calib'):
        calibs = dict(
            input_keypoints=np.linspace(0, 1, num=sizes[0]),
            output_min=0.0,
            output_max=sizes[0] - 1.0,
        )
        if any(c in mode for c in ('constrained', 'constr')):
            calibs.update(dict(
                monotonicity='decreasing',
                convexity='convex',
                kernel_regularizer=('wrinkle', 0, 1)
            ))
        calibs = [ tfl.layers.PWLCalibration(**calibs)(inputs[0]) ] + [
            tfl.layers.PWLCalibration(
                input_keypoints=np.linspace(0, 1, num=s),
                output_min=0.0,
                output_max=s - 1.0
            )(i)
            for s, i in zip(sizes[1:], inputs[1:])
        ]
        lat = tfl.layers.Lattice(
            lattice_sizes=sizes,
            output_min=0.0
        )(calibs)
        if 'out' in mode:
            lat = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0.0, 1.0, 5))(lat)
        out = [ lat ]
    elif 'dual' in mode:
        L1 = tfl.layers.Lattice(lattice_sizes=sizes)(inputs)
        L2 = tfl.layers.Lattice(lattice_sizes=sizes, output_min=0.0)(inputs)
        lat = tf.keras.layers.Concatenate(axis=-1)([ L1, L2 ])
        out = [ tf.keras.layers.Dense(1)(lat) ]
    else:
        out = [ tfl.layers.Lattice(lattice_sizes=sizes)(inputs) ]

    return tf.keras.models.Model(inputs, out)

def ProbabilisticLatticeModel(
    sizes=[12, 24],
    monotonicities=None,
    distribution='normal',
    vmin=0, vmax=1
):
    '''
    Creates a Lattice Model which predicts mean and std,
    passed then to a Distribution Lambda Layer to sample the output
    '''
    import tensorflow_lattice as tfl
    import tensorflow_probability as tfp
    DST = __get_distributions()
    inputs = [ tf.keras.layers.Input((1, )) for _ in sizes ]
    lat_mu = tfl.layers.Lattice(
        lattice_sizes=sizes,
        monotonicities=monotonicities,
        output_min=vmin,
        output_max=vmax
    )(inputs)
    lat_std = tfl.layers.Lattice(
        lattice_sizes=sizes,
        monotonicities=monotonicities,
        output_min=vmin,
        output_max=vmax
    )(inputs)

    distribution = DST.get(distribution, DST['normal'])
    lat_dist = tf.keras.layers.Concatenate(axis=-1)([ lat_mu, lat_std ])
    distr = tfp.layers.DistributionLambda(distribution)(lat_dist)
    model = tf.keras.Model(inputs, distr)
    return model

def train_model(
    model,
    x1, x2, y,
    optimizer='Adam',
    loss='mae',
    batch_size=64,
    epochs=20,
    validation_split=0.1,
    shuffle=True,
    verbose=None,
    plot=True
):
    '''
    Train the model with the given hyperparameters

    ---------
    Contextual Notes:
    - x1 = units
    - x2 = months
    - y  = price
    '''
    model.compile(optimizer=optimizer, loss=loss)
    verbose = epochs < 20 if verbose is None else verbose
    history = model.fit(
        x=(x1, x2),
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        shuffle=shuffle,
        verbose=verbose,
        callbacks=[ NanLossCallback(verbose) ]
    )
    if plot: up.plot_training_history(history, save=f'{plot}_train' if type(plot) is str else False)
    return history

def test_model(
    model,
    x1, x2, y,
    metrics=('mse', 'r2'),
    plot=True
):
    '''
    Test the model by using the given metrics
    
    ---------
    Contextual Notes:
    - x1 = units
    - x2 = months
    - y  = price'''
    predicted = model.predict((x1, x2))
    if type(predicted) is list and len(predicted) == 2:
        predicted = np.random.normal(predicted[0], predicted[1])
    if hasattr(predicted, 'numpy'): predicted = predicted.numpy()
    evaluation = um.evaluate(y, predicted, metrics=metrics)
    if plot: up.plot_pred_scatter(y, save=f'{plot}_test' if type(plot) is str else False)
    evaluation['__prediction__'] = predicted
    return evaluation