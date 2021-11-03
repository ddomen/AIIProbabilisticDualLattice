#region Program Usage
'''
usage: model.lattice.py [-h] [-i INPUT] [-x EXTENSION] [-t TARGET] [-e EPOCHS] [-b BATCH] [-s] [-c CONFIG] [-v [{critical,error,warning,info,debug,c,e,w,warn,i,d}]]

This program train a Lattice Model from an input dataset and outputs a metric evaluation of the trained model

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file (default: ./data/source.gzip)
  -x EXTENSION, --extension EXTENSION
                        the type of the file, used for sepcify the reader (eg. 'parquet' => pd.read_parquet and df.to_parquet) (default: parquet)
  -t TARGET, --target TARGET
                        The target csv file where to store the metrics output (default: metrics.csv)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 200)
  -b BATCH, --batch BATCH, --batch-size BATCH
                        Batch Size (default: 256)
  -s, --shuffle         Shuffle (default: False)
  -c CONFIG, --config CONFIG, --configuration CONFIG
                        a '.cfg' file where are stored configurations for this program (default: model.cfg)
  -v [{critical,error,warning,info,debug,c,e,w,warn,i,d}], --verbosity [{critical,error,warning,info,debug,c,e,w,warn,i,d}]
                        Provide logging level
'''
#endregion

#region Program Definition
import src.program_utils as up

args, log = up.Program(
    __file__, __name__,
    *up.common('ixtebscp', e=200, b=256),
    description='This program train a Lattice Model from an input dataset '
    ' and outputs a metric evaluation of the trained model',
    config='lattice'
)
#endregion

#region Program
log.debug('Importing modules')
import src.data_utils as ud
import src.plot_utils as uplt
import src.tf_utils as utf

if args.config.has('batch'): args.batch = args.config.getInt('batch', args.batch)
if args.config.has('epochs'): args.epochs = args.config.getInt('epochs', args.epochs)

log.info('Loading dataset')
log.debug('Reading dataset')
df = ud.as_dataframe(args.input, args.extension)

log.debug('Preprocessing')
df = ud.preprocess(df, True)
df = ud.aggregate(df)[0]

prob = args.config.getBool('prob', False)
sizes = args.config.getTuple('sizes', (16, 16), int)
splits = args.config.getTuple('splits', (0.8, 0.2), float)

if len(splits) == 1: splits = splits[0], 1.0 - splits[0]
elif len(splits) < 2: splits = 0.8, 0.2

log.debug(f'Splitting: {splits}')
df_train, df_test = ud.split_data(df, *splits, shuffle=args.shuffle)

log.debug('Generating model')
mono = None
dist = None
limits = None
mode = None
if prob:
    log.debug('Using probabilistic lattice')
    mono = args.config.getStr('monotonicity', None)
    dist = args.config.getStr('distribution', None)
    limits = args.config.getTuple('limits', (0.0, 1.0), float)
    if len(limits) == 0: limits = (0.0, 1.0)
    elif len(limits) == 1: limits = (limits[0], 1.0) 
    model = utf.ProbabilisticLatticeModel(
        sizes, mono, dist, limits[0], limits[1]
    )
    y_train = df_train['price'].values
    y_test = df_test['price'].values
else:
    log.debug('Using non-probabilistic lattice')
    mode = args.config.getStr('mode', None)
    model = utf.LatticeModel(sizes, mode=mode)
    y_train = (df_train['mean'].values, df_train['std'].values)
    y_test = (df_test['mean'].values, df_test['std'].values)

opt = args.config.getStr('optimizer', 'Adam')
loss = args.config.getStr('optimizer', 'mse')

NAME = dict(
    type='lat',
    prob=prob,
    mono=mono,
    dist=dist,
    limits=limits,
    mode=mode,
    epochs=args.epochs,
    batch=args.batch,
    shuffle=args.shuffle,
    splits=splits
)
args.plot = uplt.make_name(NAME, args.plot)

log.info('Training model')
history = utf.train_model(
    model,
    df_train['units'].values, df_train['month'].values, y_train,
    opt, loss, args.batch, args.epochs,
    shuffle=args.shuffle,
    verbose=args.isVerbose,
    plot=args.plot
)

met = args.config.getTuple('metrics', ('mse', 'r2'), str.lower)

log.info('Testing model')
pred = utf.test_model(
    model,
    df_test['units'].values, df_test['month'].values,
    y_test, metrics=met,
    plot=args.plot
)
pred['name'] = NAME

log.debug(f'Adding metrics @ {args.target}')
ud.add_metrics(args.target, pred)
log.info('All done')
#endregion