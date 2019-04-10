### implement autoencoder
from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# for reproducibility
np.random.seed(1337) 

class autoencoder(object):
    def __init__(self, encoding_dim, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.encoding_dim = encoding_dim
        
    def implement_autoencoder(self):
        # this is our input placeholder
        # input_img = Input(shape=(48,))
        input_img = Input(shape=(78,))

        # 编码层
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        # encoded = Dense(48, activation='relu')(encoded)
        encoded = Dense(78, activation='relu')(encoded)
        encoder_output = Dense(self.encoding_dim)(encoded)
        # 解码层
        decoded = Dense(10, activation='relu')(encoder_output)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        # decoded = Dense(48, activation='tanh')(decoded)
        decoded = Dense(78, activation='tanh')(decoded)
        # 构建自编码模型
        autoencoder = Model(inputs=input_img, outputs=decoded)
        # 构建编码模型
        encoder = Model(inputs=input_img, outputs=encoder_output)
        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse')
        # training
        autoencoder.fit(self.X_train, self.y_train, epochs=20, batch_size=256, shuffle=True)
        # plotting 
        encoded_imgs = encoder.predict(self.X_val)
        plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=self.y_val, s=3)
        plt.colorbar()
        plt.show()

    def lstm_autoencoder(self):
        '''below is an example of how to implement lstm autoencoder'''
        # define input sequence
        seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # reshape input into [samples, timesteps, features]
        n_in = len(seq_in)
        seq_in = seq_in.reshape((1, n_in, 1))
        # prepare output sequence
        seq_out = seq_in[:, 1:, :]
        n_out = n_in - 1
        # define model
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
        model.add(RepeatVector(n_out))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')
        plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
        # fit model
        model.fit(seq_in, seq_out, epochs=300, verbose=0)
        # demonstrate prediction
        yhat = model.predict(seq_in, verbose=0)
        print(yhat[0,:,0])


