import pandas as pd
import keras
from keras import layers

# Parameters
source_filepath = './waltz_features.csv'    # Location of Waltz features

# Read data as dataframe
waltz_df = pd.read_csv(source_filepath, sep=',', header=0)

x_train = waltz_df.drop(labels=['Sequence', 'Amyloid'], axis=1)
y_train = waltz_df.Amyloid

n_cols = x_train.shape[1]

model = keras.Sequential()
model.add(layers.Dense(1, activation='relu', input_shape=(n_cols,)))
model.add(layers.Dense(1, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

score = model.evaluate(x_train, y_train, batch_size=32)
