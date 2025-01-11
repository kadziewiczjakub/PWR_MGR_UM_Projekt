from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch, butter, lfilter
from scipy.stats import kurtosis, skew, entropy
import pandas as pd
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):
	def __init__(self, sf=128, epoch_length=1):
		self.sf = sf
		self.epoch_length = epoch_length
		self.Bands = {
			"delta": (0.5, 4),
			"theta": (4, 8),
			"alpha": (8, 13),
			"beta": (13, 30),
			"gamma": (30, 45)
		}
		
	def band_filter(self, data, low, high):
		nyquist = 0.5 * self.sf
		low = low / nyquist
		high = high / nyquist
		b, a = butter(5, [low, high], btype='band')
		return lfilter(b, a, data)
	
	def zero_crossing_rate(self, epoch):
		return ((epoch[:-1] * epoch[1:]) < 0).sum()

	def mean_absolute_deviation(self, epoch):
		return np.mean(np.abs(epoch - np.mean(epoch)))
	
	def band_power(self, data, sf, band):
		low, high = self.Bands[band]
		f, Pxx = welch(data, fs=sf, nperseg=sf*2) # spectral power density
		return np.sum(Pxx[(f >= low) & (f<=high)])
	
	def peak_frequency(self, epoch, sf, band):
		low, high = self.Bands[band]
		f, Pxx = welch(epoch, fs=sf, nperseg=sf*2)
		f_band = f[(f >= low) & (f <= high)]
		Pxx_band = Pxx[(f >= low) & (f <= high)]
		return f_band[np.argmax(Pxx_band)] if len(Pxx_band) > 0 else np.nan

	def spectral_entropy(self, epoch, sf):
		f, Pxx = welch(epoch, fs=sf, nperseg=sf*2)
		Pxx /= np.sum(Pxx)
		return -np.sum(Pxx * np.log2(Pxx + 1e-12)) 
	
	def hurst_exponent(self, signal):
		N = len(signal)
		T = np.arange(1, N + 1)
		Y = np.cumsum(signal - np.mean(signal))
		R = np.max(Y) - np.min(Y)
		S = np.std(signal)
		return np.log(R / S) / np.log(N) if S != 0 else 0.5 
	
	def time_domain_features(self, epoch):
		return {
			"mean": np.mean(epoch),
			"std_dev": np.std(epoch),
			"rms": np.sqrt(np.mean(epoch ** 2)),
			"kurtosis": kurtosis(epoch),
			"skewness": skew(epoch),
			"entropy": entropy(np.histogram(epoch, bins=10)[0]),
			"zero_crossing_rate": self.zero_crossing_rate(epoch),
			"mean_absolute_deviation": self.mean_absolute_deviation(epoch)
		}

	def frequency_domain_features(self, epoch, sf):
		band_powers = {f"{band}_power": self.band_power(epoch, sf, band) for band in self.Bands}
		total_power = sum(band_powers.values())
		relative_powers = {f"{band}_relative_power": power / total_power for band, power in band_powers.items()}

		peak_frequencies = {f"{band}_peak_freq": self.peak_frequency(epoch, sf, band) for band in self.Bands}
		spectral_entropy_feature = {"spectral_entropy": self.spectral_entropy(epoch, sf)}

		return {**band_powers, **relative_powers, **peak_frequencies, **spectral_entropy_feature}

	def nonlinear_features(self, epoch):
		return {
			#"approximate_entropy": self.approximate_entropy(epoch), #takes too long
			"hurst_exponent": self.hurst_exponent(epoch)
		}

	def extract_epoch_features(self, epoch):
		filtered_data = self.band_filter(epoch, 0.5, 45)

		features = self.time_domain_features(filtered_data)
		freq_features = self.frequency_domain_features(filtered_data, self.sf)
		non_linear_features = self.nonlinear_features(filtered_data)

		features.update(freq_features)
		features.update(non_linear_features)

		return features
	
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		"""Extract features from EEG data.
		X: list of tuples -> (data, arousal, valence, subject, game)
		data -> array with channel data
		"""
		
		feature_list = []
		data = X['data']
		n_epochs = data.shape[0]
		print(f"S{X['subject']} G{X['game']} - Extracting features from {n_epochs} epochs")
		for epoch in range(n_epochs):
			epoch_data = data[epoch]
			n_channels = epoch_data.shape[0]
			for channel in range(n_channels):
				channel_data = epoch_data[channel]
				features = self.extract_epoch_features(channel_data)
				features['channel'] = channel
				features['valence'] = X['valence']
				features['arousal'] = X['arousal']
				feature_list.append(features)
		return pd.DataFrame(feature_list)
