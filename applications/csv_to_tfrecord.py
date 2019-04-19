#!python
import os
import tensorflow as tf
from tensorflow.data.experimental import make_csv_dataset

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
        'file_pattern',
        None,
        ('List of files or patterns of file paths containing CSV records.'
        'See tf.gfile.Glob for pattern rules.')
)
flags.mark_flag_as_required('file_pattern')

flags.DEFINE_integer(
        'batch_size',
        32,
        'An int representing the number of records to combine in a single batch.',
        lower_bound=0
)

flags.DEFINE_list(
        'column_names',
        None,
        ('An optional list of strings that corresponds to the CSV columns, in order. '
        'One per column of the input record.')
)

flags.DEFINE_list(
        'column_defaults',
        None,
        ('A optional list of default values for the CSV fields. '
        'One item per selected column of the input record.')
)

flags.DEFINE_list(
        'label_name',
        None,
        'A optional string corresponding to the label column.'
)

flags.DEFINE_list(
    'select_columns',
    None,
    ('An optional list of integer indices or string column names, '
    'that specifies a subset of columns of CSV data to select.')
)

flags.DEFINE_string(
    'field_delim',
    ',',
    'Char delimiter to separate fields in a record.'
)

flags.DEFINE_bool(
    'use_quote_delim',
    True,
    ('If false, treats double quotation marks as regular characters inside of '
    'the string fields.')
)

flags.DEFINE_string(
    'na_value',
    '',
    'Additional string to recognize as NA/NaN.'
)

flags.DEFINE_bool(
    'header',
    True,
    ('A bool that indicates whether the first rows of provided CSV files correspond to '
    'header lines with column names, and should not be included in the data.')
)

flags.DEFINE_integer(
    'num_epochs',
    1,
    ('An int specifying the number of times this dataset is repeated. '
    'If None, cycles through the dataset forever. Defaults to 1.')
)

flags.DEFINE_bool(
    'shuffle',
    True,
    'A bool that indicates whether the input should be shuffled.'
)

flags.DEFINE_integer(
    'shuffle_buffer_size',
    10000,
    ('Buffer size to use for shuffling.'
    'A large buffer size ensures better shuffling, but increases '
    'memory usage and startup time.')
)

flags.DEFINE_integer(
    'shuffle_seed',
    None,
    'Randomization seed to use for shuffling.'
)

#flags.DEFINE_integer(
    #'prefetch_buffer_size',
    #tf.data.optimization.AUTOTUNE,
    #"""An int specifying the number of feature batches to prefetch for performance improvement.
    #Recommended value is the number of batches consumed per training step.
    #Defaults to auto-tune."""
#)

flags.DEFINE_integer(
    'num_parallel_reads',
    1,
    ('Number of threads used to read CSV records from files. '
    'If >1, the results will be interleaved.')
)

flags.DEFINE_bool(
    'sloppy',
    False,
    ('If True, reading performance will be improved at the cost of '
    'non-deterministic ordering. '
    'If False, the order of elements produced is deterministic prior to '
    'shuffling (elements are still randomized if shuffle=True).')
)

flags.DEFINE_integer(
    'num_rows_for_inference',
    100,
    ('Number of rows of a file to use for type inference if record_defaults '
    'is not provided. '
    'If None, reads all the rows of all the files.')
)

flags.DEFINE_string(
    'compression_type',
    None,
    ('String evaluating to one of "" (no compression), '
    '"ZLIB", or "GZIP".')
)

flags.DEFINE_bool(
    'ignore_errors',
    False,
    ('If True, ignores errors with CSV file parsing, such as malformed data or '
    'empty lines,and moves on to the next valid CSV record. '
    'Otherwise, the dataset raises an error and stops processing when encountering '
    'any invalid records.')
)

def main(argv):

    ds = make_csv_dataset(
        file_pattern=FLAGS.file_pattern,
        batch_size=FLAGS.batch_size,
        column_names=FLAGS.column_names,
        column_defaults=FLAGS.column_defaults,
        label_name=FLAGS.label_name,
        select_columns=FLAGS.select_columns,
        field_delim=FLAGS.field_delim,
        use_quote_delim=FLAGS.use_quote_delim,
        na_value=FLAGS.na_value,
        header=FLAGS.header,
        num_epochs=FLAGS.num_epochs,
        shuffle=FLAGS.shuffle,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        shuffle_seed=FLAGS.shuffle_seed,
        #prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        num_parallel_reads=FLAGS.num_parallel_reads,
        sloppy=FLAGS.sloppy,
        num_rows_for_inference=FLAGS.num_rows_for_inference,
        compression_type=FLAGS.compression_type,
        ignore_errors=FLAGS.ignore_errors
    ).take(1)

    for x in ds:
        print(x)



if __name__ == '__main__':
  app.run(main)
