import pandas as pd
import keras
from keras import layers
from keras.activations import relu, elu, tanh, linear
from sklearn.model_selection import train_test_split
import talos

# Parameters
source_filepath = './waltz_features.csv'    # Location of Waltz features

# Read data as dataframe
waltz_df = pd.read_csv(source_filepath, sep=',', header=0)

X = waltz_df.drop(labels=['Sequence', 'Amyloid'], axis=1)
Y = waltz_df.Amyloid

n_cols = X.shape[1]

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def waltz_model(x_train, y_train, x_target, y_target, params):
    model = keras.Sequential()
    model.add(layers.Dense(params['first_neuron'], activation=params['activation'], input_shape=(n_cols,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['accuracy'])
    out = model.fit(x_train, y_train, epochs=params['epochs'], validation_data=[x_target, y_target],
                    batch_size=params['batch_size'])
    return out, model


params = {
    'activation': [relu, tanh, linear],
    'first_neuron': [12, 24],
    'optimizer': ['Adam'],
    'losses': ['binary_crossentropy'],
    'hidden_layers': [0, 1, 2],
    'batch_size': [20, 30, 40],
    'epochs': [10, 100],
}

scan = talos.Scan(x_train.values, y_train.values, model=waltz_model, params=params)

scan.data.head()
