_log_alias = dict(
    c='critical',
    e='error',
    w='warning', warn='warning',
    i='info',
    d='debug'
)
_log_levels = (
    'critical', 'error', 'warning',
    'info', 'debug'
)

class Section(object):
    '''Easily reads and manage sections from .cfg files'''
    HIDDEN_ACCESSOR = '..section..'
    def __init__(self, section):
        super().__setattr__(Section.HIDDEN_ACCESSOR, dict(section))

    def __try(self, v, t, f=None):
        if not callable(t): return v
        try: return t(v)
        except: return f

    def __bool(self, value): return value.lower() in ('yes', 'y', 'true', 'on', '1') if type(value) is str else (not not value)
    def __eval(self, value): return eval(value) if type(value) is str else value
    def __iter(self, value, t, f=None):
        if type(value) is str: value = value.split(',')
        if type(value) in (list, tuple, set):
            return [ self.__try(x, t, f) for x in value ]
        return []

    def __getattr__(self, attr): return self.get(attr)
    def __setattr__(self, attr, value): return self.set(attr, value)

    def __getitem__(self, index, c=None, f=None):
        t = type(index)
        if t in (tuple, list, set):
            return t(self.__getitem__(i, c, f) for i in index)
        elif t is dict:
            return { k: self.__getitem__(v, c, f) for k, v in index.items() }
        elif type(index) is slice:
            return self.__getitem__(index.start, index.stop, index.step)
        else:
            return self.get(index, f, c)

    def __setitem__(self, index, value, c=None, f=None):
        ti = type(index)
        if ti in (tuple, list, set):
            for i in index: self.__setitem__(i, value, c, f)
        elif ti is dict:
            for k, v in index: self.__setitem__(k, self.__try(value, v), c, f)
        elif ti is slice:
            return self.__setitem__(index.start, self.__try(value, index.stop, index.step))
        else: return self.set(index, value)
    
    def has(self, index):
        index = str(index).lower()
        return index in super().__getattribute__(Section.HIDDEN_ACCESSOR)

    def get(self, index, fallback=None, type=None):
        index = str(index).lower()
        result = super().__getattribute__(Section.HIDDEN_ACCESSOR).get(index, fallback)
        result = self.__try(result, type, fallback)
        return result
    
    def set(self, index, value):
        index = str(index).lower()
        super().__getattribute__(Section.HIDDEN_ACCESSOR)[index] = value
        return self

    def getInt(self, index, fallback=0): return self.get(index, fallback, int)
    def getFloat(self, index, fallback=0.0): return self.get(index, fallback, float)
    def getBool(self, index, fallback=False): return self.get(index, fallback, self.__bool)
    def getStr(self, index, fallback=''): return self.get(index, fallback, str)
    def getEval(self, index, fallback=None): return self.get(index, fallback, self.__eval)
    def getList(self, index, fallback=None, type=None): return list(self.get(index, fallback, lambda v: self.__iter(v, type)))
    def getTuple(self, index, fallback=None, type=None): return tuple(self.getList(index, fallback, type))
    def getSet(self, index, fallback=None, type=None): return set(self.getList(index, fallback, type))

    def __lshift__(self, other):
        print(self, '<<', other)
        return 0

    def __str__(self): return f'Section<{super().__getattribute__("...")}>'
    __repr__ = __str__

def Program(
    file, name, *args,
    description=None, show_args=True,
    only_program=True,
    format='[%(asctime)s] %(message)s',
    date_format='%m/%d/%Y %H:%M:%S',
    config=None
):
    '''
    Define a program with cli arguments and a logger.
    You should pass `__file___` and `__name__` as first two arguments
    '''
    if only_program: _throw_if_module(name)
    args = _get_args(*args, description=description)
    log = _get_logger(
        file, args.verbosity,
        format=format, date_format=date_format
    )
    if show_args: _log_args(file, args, log.info)
    if hasattr(args, 'config') and type(args.config) is str and (args.config.endswith('.cfg') or args.config.endswith('.py')):
          args.config = _get_config(args.config, config)  
    return args, log

def _get_config(path, section):
    '''Retrieve and read config file from cli arguments'''
    if path.endswith('.cfg'):
        import configparser
        try:
            config = configparser.ConfigParser()
            config.read(path)
            if section is not None:
                if not config.has_section(section):
                    section = str(section).lower()
                    for s in config.sections():
                        if s.lower() == section:
                            section = s
                            break
                if config.has_section(section): config = config[section]
                else: config = config['DEFAULT']
                config = Section(config)
            else:
                config = Section({ k: Section(v) for k, v in config.items() })
            return config
        except: return Section({})
    else:
        try:
            module = {}
            with open(path, 'r') as f:
                exec(f.read(), { 'Section': Section }, module)
            if 'config' in module:
                module = module['config']
                if callable(module): 
                    try: return module(section)
                    except:
                        try: return module()
                        except: return module
            return module
        except: return None

def _get_args(*args, description=None):
    '''Get the parsed arguments from the cli'''
    import argparse
    ap = argparse.ArgumentParser(description=description)
    for args in args: ap.add_argument(*args[:-1], **args[-1])
    ap.add_argument(
        '-v', '--verbosity',
        nargs='?',
        type=str.lower, default='info',
        choices=list(_log_levels) + list(_log_alias),
        help='Provide logging level (default: info)'
    )
    ap.add_argument(
        '-B', '-st', '--trill', '--sound-trill', '--beep',
        action='store_const',
        const=True,
        default=False,
        help='Produce a sound when the program terminates (default: False)'
    )
    args = ap.parse_args()
    if type(args.verbosity) is not str: args.verbosity = 'debug'
    args.verbosity = _log_alias.get(args.verbosity, args.verbosity)
    args.isVerbose = args.verbosity == 'debug'
    if hasattr(args, 'plot'):
        args.plot = args.plot if len(args.plot or '') else True
    else: args.plot = False
    if args.trill: __beep_on_exit()
    return args

def _get_logger(name, verbosity, format, date_format):
    '''Retrieve the logger for the current program'''
    import logging
    logging.basicConfig(
        format=format,
        datefmt=date_format
    )
    if type(verbosity) is str:
        verbosity = verbosity.upper()
        if hasattr(logging, verbosity):
            verbosity = getattr(logging, verbosity)
        else: verbosity = logging.INFO
    log = logging.getLogger(name)
    log.setLevel(verbosity)
    return log

def _cammel_name(name, separator='.', join=' '):
    '''
    Transform a name in a cammel case version.
    By default split the name every '.' and rejoin it with a space:

    eg. `alpha.beta -> Alpha Beta`
    '''
    return str.join(join, (x[0].upper() + x[1:] for x in name.split(separator)))

def _log_args(name, args, logger):
    '''Pretty log of the arguments'''
    name = name.lower()
    if name.endswith('.py'): name = name[:-3]
    name = _cammel_name(name)
    eq = '=' * len(name)
    args = dict(vars(args))
    args_template = ''
    args = { f'{_cammel_name(k)}': v for k, v in args.items() }
    max_key_len = max([ len(k) for k in args ]) + 5
    for k, v in args.items():
        tab = ' ' * (max_key_len - len(k))
        args_template += f'{_cammel_name(k)}:{tab}{v}\n'
    logger(f'''
================{eq}
===== AII {name} =====
================{eq}

===== Arguments =====
{args_template[:-1]}
=====================
''')

def _throw_if_module(name):
    '''
    Check if the name of the module is a program,
    if not raise an exception
    '''
    if name != '__main__':
        raise Exception('This file is a program, you should not use it as a module')

_common_values = dict(
    input=(
        '-i', '--input',
        dict(
            type=str,
            default='./data/data.gzip',
            help='input file'
        )
    ),
    output=(
        '-o', '--output',
        dict(
            type=str,
            default='./data/data.gzip',
            help='the output file'
        )
    ),
    extension=(
        '-x', '--extension',
        dict(
            type=str.lower,
            default='parquet',
            help='the type of the file, used for sepcify the reader '
            '(eg. \'parquet\' => pd.read_parquet and df.to_parquet)'
        )
    ),
    target=(
        '-t', '--target',
        dict(
            type=lambda s: s if s.endswith('.csv') else (s + '.csv'),
            default='metrics.csv',
            help='The target csv file where to store the metrics output'
        )
    ),
    epochs=(
        '-e', '--epochs',
        dict(
            type=int,
            default=200,
            help='Number of epochs'
        )
    ),
    batch_size=(
        '-b', '--batch', '--batch-size',
        dict(
            type=int,
            default=256,
            help='Batch Size'
        )
    ),
    shuffle=(
        '-s', '--shuffle',
        dict(
            action='store_const',
            const=True,
            default=False,
            help='Shuffle'
        )
    ),
    plot=(
        '-p', '--plot',
        dict(
            type=str,
            nargs='?',
            default=None,
            help='The directory where to save plots. '
            'If the parameter is present but it is left void, '
            'the plots are shown in a window'
        )
    ),
    configuration=(
        '-c', '--config', '--configuration',
        dict(
            type=lambda s: s if s.endswith('.cfg') or s.endswith('.py') else (s + '.cfg'),
            default='model.cfg',
            help='a \'.cfg\' file where are stored configurations for this program'
        )
    )
)

_common_alias = dict(
    i='input',
    o='output',
    x='extension',
    t='target',
    e='epochs',
    b='batch_size', batch='batch_size',
    s='shuffle',
    c='configuration', config='configuration',
    p='plot'
)

def _join_dicts(dest, *sources):
    '''Merge two or more dicts into `dest`'''
    for s in sources:
        if not s: continue
        dest.update(s)
    return dest

def common(*args, **default):
    '''
    Retrieve frequently used arguments by their name.
    It is possible to use only one argument with a
    string of the initial letters of each common argument
    '''
    if len(args) == 1 and _common_alias.get(args[0], args[0]) not in _common_values:
        args = list(args[0])
    default = { _common_alias.get(k, k): v for k, v in default.items() }
    result = []
    for arg in args:
        d = None
        if arg in default: d = dict(default=default.get(arg))
        arg = _common_alias.get(arg, arg)
        if arg not in _common_values: continue
        if d is None and arg in default: d = dict(default=default.get(arg))
        arg = _common_values.get(arg)
        arg = tuple(arg[:-1]) + (_join_dicts({}, arg[-1], d),)
        arg[-1]['help'] = f'{arg[-1].get("help", "")} (default: {arg[-1].get("default", None)})'
        result.append(arg)
    return tuple(result)

def parse_configuration(arg, parser):
    '''
    Parse a configuraion from a string.
    The `parser` argument can be a `list`/`tuple`/`set` of
    keys to map the arguments or a `dict` with keys of the
    configuration as keys and a mapper function as values.
    If the values are not callable the string argument is given.
    '''
    args = [ (x or '').strip() for x in arg.split(',') ]
    if type(parser) in (list, tuple, set):
        return { k: a for k, a in zip(parser, args) }
    elif type(parser) is dict:
        return { k: v(a) if callable(v) else a for (k, v), a in zip(parser.items(), args) }
    return args

def configuration(
    shorthand='-ic',
    option='--inline-configuration',
    default={},
    help='A coma separated set of configuration with the following order: [ {ordered} ]',
    **parser
):
    '''
    Creates a wrapper for `parse_confinguration` function,
    useful to be passed as a cli argument
    '''
    return (
        shorthand, option,
        dict(
            default=default,
            type=lambda arg: parse_configuration(arg, parser),
            help=help.format(ordered=str.join(', ', parser.keys()))
        )
    )

def __beep_on_exit():
    '''
    Register the handler that produces an acustic signal
    when the process has ended (should works only on Windows OS).
    '''
    import atexit
    def _beep():
        import winsound
        winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
    atexit.register(_beep)