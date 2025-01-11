from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import os

class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, subjects=range(1, 29), games=range(1, 5), eeg_path='./GAMEEMO', sam_path='./SAMResults.csv', split_x_y=False, sf=128, window_length=1, overlap=0.5):
        self.subjects = subjects
        self.games = games
        self.eeg_path = eeg_path
        self.sam_path = sam_path
        self.split_xy = split_x_y
        self.sf = sf
        self.window_length=window_length
        self.sam_ratings = {}
        self.loaded_raw_data = None
        self.loaded_windowed_data = None
        self.overlap = overlap

    def fit(self, X=None, y=None):
        self.sam_ratings = self.load_sam_ratings()
        self.loaded_raw_data = self.load_all_data()
        return self

    def transform(self, X=None):
        if X is not None:
            self.loaded_raw_data = X
        if self.loaded_raw_data is None:
            self.fit()

        
        windowed_data = []

        tmp = self.loaded_raw_data

        for entry in tmp:
            data = entry['data']
            arousal, valence = entry['arousal'], entry['valence']

            
            window = self.split_data_into_windows(data, self.overlap)
            
            tmp_dict = {'data': window, 'subject': entry['subject'], 'game': entry['game']}
            if arousal is not None and valence is not None:
                tmp_dict['arousal'] = arousal
                tmp_dict['valence'] = valence
            windowed_data.append(tmp_dict)

        self.loaded_windowed_data = windowed_data

        if self.split_xy:
            return self.split_x_y(self.loaded_windowed_data)
        
        return self.loaded_windowed_data

    def load_sam_ratings(self):
        """Load SAM ratings into a dictionary."""
        sam_df = pd.read_csv(self.sam_path)
        self.sam_ratings = {row['subject_game']: (row['valence'], row['arousal']) for _, row in sam_df.iterrows()}
        return self.sam_ratings

    def get_sam_rating(self, subject, game):
        """Get SAM rating for a specific subject and game."""
        key = f"S{subject:02d}G{game}"
        return self.sam_ratings.get(key, (None, None))

    def load_eeg_data(self, subject, game):
        """Load EEG data for a specific subject and game."""
        subject_path = os.path.join(self.eeg_path, f"(S{str(subject).zfill(2)})/Preprocessed EEG Data/.csv format")
        file_path = os.path.join(subject_path, f"S{str(subject).zfill(2)}G{game}AllChannels.csv")
        eeg_df = pd.read_csv(file_path).iloc[:, :-1]  # Remove the last empty column
        return eeg_df.to_numpy().T

    def split_data_into_windows(self, data, overlap=0.5):
        window_size = int(self.sf * self.window_length)
        step_size = int(window_size * (1 - overlap))
        num_windows = (data.shape[1] - window_size) // step_size + 1

        windows = [data[:, i * step_size:i * step_size + window_size] for i in range(num_windows)]
        return np.array(windows)

    def load_all_data(self):
        """Load EEG and SAM data across subjects and games."""
        all_data = []
        for subject in self.subjects:
            for game in self.games:
                data = self.load_eeg_data(subject, game)
                arousal, valence = self.get_sam_rating(subject, game)
                #add raw data to list
                all_data.append({'data': data, 'arousal': arousal, 'valence': valence, 'subject': subject, 'game': game})

        return all_data

    def split_x_y(self, data):
        return ([entry['data'] for entry in data], 
                ([entry['arousal'] for entry in data], 
                 [entry['valence'] for entry in data]))
