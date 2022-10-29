import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model



class Autoencoder(Model):
    def __init__(self):
        super(self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="sigmoid"),
            layers.Dense(512, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderCP(Autoencoder):
    # Autoencoder for chromatogram de-noising and generation of consensus peak
    ''' Autoencoder training for chromatogram de-noising and generation of consensus peaks '''
    def __init__(self, rec, optim='adam', loss='mae', epos=100, bsize=20):
        self.model = super(self).__init__()
        self.model.compile(optimizer=optim, loss=loss)
        self.recTf= tf.convert_to_tensor(rec)
        self.pars = dict(optim=optim, loss=loss, epos=epos, bsize=bsize)

    def train(self):
        self.trainingEvol = self.model.fit(self.recTf, self.recTf, epochs=self.pars['epos'], batch_size=self.pars['bsize'], shuffle=True)

        self.enc_rec = self.model.encoder(self.recTf).numpy()
        self.dec_rec = self.model.decoder(self.enc_rec).numpy()
        self.consP = np.mean(self.dec_rec, 0)

    def visTrainingError(self, compLabel=None):
        fig, ax = plt.subplots()

        ax.figure(figsize=(5, 4))
        ax.plot(self.trainingEvol["loss"], label="Training Loss")
        if compLabel is not None:
            ax.set_title(compLabel)
        ax.set_xlabel(f'Epoch (batch size {self.pars["epos"]})')
        ax.set_ylabel('Mean absolute error')
        ax.legend()

    def visConsensus(self):
        recMean = np.mean(self.recTf, 0)
        recSd = np.std(self.recTf, 0)

        fig, ax = plt.subplots()
        ax.plot(self.dec_rec.T, color='black', linewidth=0.1)
        ax.plot(self.dec_rec.T, color='black', linewidth=0.1, label='AE reconstructed signal')
        ax.plot(recMean, color='yellow', linewidth=2, label='Avg signal')
        ax.plot(recMean + recSd, color='cyan', linewidth=2, label='Avg signal +/- 1 SD')
        ax.plot(recMean - recSd, color='cyan', linewidth=2)
        ax.plot(self.consP, color='red', linewidth=2, label='AE consensus peak')
        ax.set_xlabel('Normalised ST')
        ax.set_ylabel('Normalised sum of counts')
        ax.legend()
        fig.show()

