""" This module contains the pipeline train method
"""

from mnist_cnn.model import build_model, generate_model_config
from mnist_cnn.batch_gen import pipeline_batch_gen, generate_batch_spec

from keras.optimizers import Adam

def generate_train_config(model_spec, train_batch_spec, val_batch_spec,  epoch_size,
                        n_epochs=100, lr=0.001):
    """ Generates a spec fully specifying a run of the experiment

    Args:
      model_spec: a model spec
      train_batch_spec: a batch spec for train
      val_batch_spec: batch spec for validation
      epoch_size: int
      n_epochs: int
      lr: learning rate - float

    Returns:
      train_spec
    """

    train_spec = {
        'model_spec': model_spec,
        'train_batch_spec': train_batch_spec,
        'val_batch_spec':  val_batch_spec,
        'train_spec': {
            'epoch_size': epoch_size,
            'n_epochs': n_epochs,
            'lr': lr
        }
    }
    return train_spec

def pipeline_train(train_spec):
    """ Pipelined train method that trains model according to the spec
    """
    model = build_model(train_spec['model_spec'])
    # compile model loss
    optimizer = Adam(lr=train_spec['train_spec']['lr'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    train_batch_gen = pipeline_batch_gen(train_spec['train_batch_spec'])
    val_batch_gen = pipeline_batch_gen(train_spec['val_batch_spec'])


    model.fit_generator(train_batch_gen, train_spec['train_spec']['epoch_size'],
                        epochs=train_spec['train_spec']['n_epochs'],
                        validation_data=val_batch_gen, validation_steps=128)

def pipeline_train_hook(name, n_filters, nonlinearity='relu', batch_size=32,
                         epoch_size=256, n_epochs=100, lr=0.001):
    # plumb args into old config methods, get them into train_spec format
    model_spec = generate_model_config(name, n_filters, nonlinearity)
    train_batch_spec = generate_batch_spec('train',  batch_size)
    val_batch_spec = generate_batch_spec('val',  batch_size)

    train_spec = generate_train_config(model_spec, train_batch_spec, val_batch_spec,
                                       epoch_size=epoch_size, n_epochs=n_epochs, lr=lr)
    pipeline_train(train_spec)
