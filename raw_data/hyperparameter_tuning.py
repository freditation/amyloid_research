import pandas as pd
import keras
from keras import layers
from keras.activations import relu, elu, tanh, linear
import talos

# Parameters
source_filepath = './waltz_features.csv'    # Location of Waltz features

# Read data as dataframe
waltz_df = pd.read_csv(source_filepath, sep=',', header=0)

x = waltz_df.drop(labels=['Sequence', 'Amyloid'], axis=1)
y = waltz_df.Amyloid

n_cols = x.shape[1]


def waltz_model(x_train, y_train, x_target, y_target, params):
    model = keras.Sequential()
    model.add(layers.Dense(params['first_neuron'], activation=params['activation'], input_shape=(n_cols,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['accuracy'])
    out = model.fit(x_train, y_train, epochs=params['epochs'], validation_data=[x_target, y_target],
                    batch_size=params['batch_size'])
    return out, model


params = {
    'activation': [relu, elu, tanh, linear ],
    'first_neuron': [12, 24, 48],
    'optimizer': ['Nadam', 'Adam'],
    'losses': ['binary_crossentropy', 'logcosh'],
    'hidden_layers': [0, 1, 2],
    'batch_size': [20, 30, 40],
    'epochs': [10, 100, 200],
}

talos.Scan(x.values, y.values, model=waltz_model, params=params)
