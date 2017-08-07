""" This module contains methods for generating batches of data
"""

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def generate_batch_spec(mode, batch_size):
    """ Generates a spec describing how to draw batches

    Args:
      mode: one of ['train', 'test', 'val']
    """
    assert mode in ['train', 'test', 'val']

    # on a more complicated dataset this would include useful arguments
    # such as whether to augment the data, what data source to draw from....
    # this example is simple enough to not need a spec so this is just for illustration purposes
    batch_spec = {
        'mode': mode,
        'batch_size': batch_size
    }
    return batch_spec


def pipeline_batch_gen(batch_spec):
    """ Pipelined batch_gen that generates batches forever according to the spec

    Args:
      batch_spec
    """
    print("Loading dataset...")
    mnist = read_data_sets('MNIST_data', one_hot=True)

    mode = batch_spec['mode']
    batch_size = batch_spec['batch_size']

    while True:
        if mode == 'train':
            yield mnist.train.next_batch(batch_size)
        if mode == 'test':
            yield mnist.test.next_batch(batch_size)
        if mode == 'val':
            yield mnist.val.next_batch(batch_size)
