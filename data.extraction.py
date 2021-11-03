#region Program Usage 
'''
usage: data.extraction.py [-h] [-i INPUT] [-e EXTENSION] [-o OUTPUT]
                          [-v [{critical,error,warning,info,debug,c,e,w,warn,i,d}]]

This program will preprocess the full Dataset selecting just a minimal part of
it. For huge dataset it will take some time to process

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file
  -e EXTENSION, --extension EXTENSION
                        the type of the file, used for sepcify the reader (eg.
                        'parquet' => pd.read_parquet and df.to_parquet)
  -o OUTPUT, --output OUTPUT
                        the output file
  -v [{critical,error,warning,info,debug,c,e,w,warn,i,d}], --verbosity [{critical,error,warning,info,debug,c,e,w,warn,i,d}]
                        Provide logging level
'''
# endregion

#region Program Definition
import src.program_utils as up
args, log = up.Program(
    __file__, __name__,
    *up.common('ixo', i='./data/source.gzip'),
    description='This program will preprocess the full Dataset '
        'selecting just a minimal part of it. '
        'For huge dataset it will take some time to process'
)
# endregion

#region Program
log.debug('Importing modules')
import src.data_utils as ud

### load the dataset
log.info('Loading the dataset')
df = ud.as_dataframe(args.input, args.extension).reset_index()
log.debug('Done')

### select the most important product
log.info('Processing')
log.debug('Selecting the most present product (count of dates)')
brands = df.groupby('brandcode').agg(dict(TripDate='count'))['TripDate']
target_brand = float(brands.idxmax())
log.debug(f'Selected {target_brand:0.0f}')

### normalizing price inter-product
log.debug('Inter-product normalization')
df.loc[:, 'PRICE'] = df['Amount'] / df['Units']
df = ud.intra_class_norm(df, 'PRODUCT_ID', target='PRICE', input='Amount', output='Amount')

### select only intereseted columns
log.debug('Filtering columns and renaiming [TripDate, Units, Amount] -> [date, units, amount]')
rf = df.loc[df['brandcode'] == target_brand, ['TripDate', 'Units', 'Amount']].rename(columns=dict(
    TripDate='date',
    Units='units',
    Amount='amount'
))


### save the dataframe
log.info(f'Saving the results @ {args.output}')
ud.save_dataframe(rf, args.output, args.extension)

### Done
log.info('All done')

#endregion