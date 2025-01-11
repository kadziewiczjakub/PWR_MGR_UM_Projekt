import os
import pickle
from data_loader import DataLoader
from feature_extractor import FeatureExtractor

SF = 128 #sampling frequency

#save feature extraction data
def load_data_save_extracted_features(windows, overlaps):
	for window in windows:
		for overlap in overlaps:
			print(f"Extracting features for {window} seconds windows with {overlap} overlap")
			path = f'all_prepared_data_{window}s_overlap_{int(overlap*100)}.pkl'
			if os.path.exists(path):
				pass
			else:
				data_loader = DataLoader(subjects=range(1, 29), games=range(1, 5), eeg_path='./GAMEEMO', sam_path='./SAMResults.csv', split_x_y=False, sf=SF, window_length=window, overlap=overlap)
				feature_extractor = FeatureExtractor()
				all_data = data_loader.fit_transform(None)
				all_features = [feature_extractor.transform(sample) for sample in all_data]
				all_prepared_data = list(zip(all_data, all_features))

				pickle.dump(all_prepared_data, open(path, 'wb'))

if __name__ == '__main__':
	windows = [2, 3, 10, 20, 60, 120] # TODO change if needed
	overlaps = [0, 0.25, 0.5, 0.75] # TODO change if needed
	load_data_save_extracted_features(windows, overlaps)