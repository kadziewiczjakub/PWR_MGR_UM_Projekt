import os
import pickle
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from exp_utils import load_data_save_extracted_features
from copy import deepcopy

def save_checkpoint(state, CHECKPOINT_FILE):
	with open(CHECKPOINT_FILE, "wb") as f:
		pickle.dump(state, f)


def load_checkpoint(CHECKPOINT_FILE):
	if os.path.exists(CHECKPOINT_FILE):
		with open(CHECKPOINT_FILE, "rb") as f:
			return pickle.load(f)
	return None

def prepare_data_for_exp3(prepared_data, time_frame):
	start_percent, end_percent = time_frame

	start_index = int(max(0, start_percent * prepared_data[0][1].shape[0] / 100))
	end_index = int(min(prepared_data[0][1].shape[0], end_percent * prepared_data[0][1].shape[0] / 100))

	train_data = []
	test_data = []

	train_f = [data[1][start_index:end_index] for data in prepared_data]
	test_f = [data[1][:start_index] for data in prepared_data] + [data[1][end_index:] for data in prepared_data]

	return pd.DataFrame(train_data), pd.DataFrame(test_data), pd.concat(train_f), pd.concat(test_f)

def exp3():
	CHECKPOINT_FILE = "exp3_checkpoint.pkl"
	#windows = [2, 3, 6, 10, 20, 30, 60, 90, 120]
	windows = [2, 3, 10, 20, 60, 120] # TODO change if needed
	overlaps = [0, 0.25, 0.5, 0.75] # TODO change if needed
	sf = 128
	train_time_frames = [(0, 10), (10, 20), (20, 30), 
						 (30, 40), (40, 50), (50, 60),
						 (60, 70), (70, 80), (80, 90), (90, 100), 
						 (0,20), (20,40), (40,60), (60,80),
						(0,30), (30,60), (0,60), (0,90)]
	N_REPEATS = 1 # TODO change
	SUBJECTS = list(range(1, 29))
	GAMES = list(range(1,5))

	CLASSIFIERS = [
		RandomForestClassifier(n_estimators=100, n_jobs=-1),
		RandomForestClassifier(n_estimators=500, n_jobs=-1),
		KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
		#HistGradientBoostingClassifier(max_iter=1000),
		#SVC()
	]

	SPLITTERS = [
		KFold(n_splits=5, shuffle=True),
		RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
		#StratifiedKFold(n_splits=5, shuffle=True),
	]

	checkpoint = load_checkpoint(CHECKPOINT_FILE)
	if checkpoint:
		f_all_reports = checkpoint["f_all_reports"]	#features reports
		savepoint_keys = checkpoint["savepoint_keys"]
		print("Resuming from checkpoint...")
	else:
		f_all_reports = []
		savepoint_keys = {}
	for repeat in range(N_REPEATS):
		for overlap in overlaps:
			for classifier_id, clf_template in enumerate(CLASSIFIERS):
				for splitter_id, splitter in enumerate(SPLITTERS):
					for window in windows:
						savepoint_key = f'{repeat}_{classifier_id}_{splitter_id}_{window}_{overlap}'
						if savepoint_key in savepoint_keys:
							continue
						path = f'all_prepared_data_{window}s_overlap_{int(overlap*100)}.pkl'
						all_prepared_data = pickle.load(open(path, 'rb'))
					
						for time_frame_idx, time_frame in enumerate(train_time_frames):
							savepoint_key = f'{repeat}_{classifier_id}_{splitter_id}_{window}_{overlap}'
							#if time_frame_idx < start_frame:
							#	continue
							print(f"Classifier: {classifier_id}, Splitter: {splitter_id}, Window: {window}, Overlap: {overlap}, time_frame: {time_frame}, Repeat: {repeat}")

							_, _, train_f, test_f = prepare_data_for_exp3(all_prepared_data, time_frame)

							X = train_f.drop(columns=['arousal', 'valence', 'subject', 'game'], errors='ignore')
							Y = pd.DataFrame(train_f[['arousal', 'valence']])
							Y = pd.DataFrame({'arousal': np.where(Y['arousal'] >= 0, 1, 0), 'valence': np.where(Y['valence'] >= 0, 1, 0)})
							
							eval_X = test_f.drop(columns=['arousal', 'valence', 'subject', 'game'], errors='ignore')
							eval_Y = pd.DataFrame(test_f[['arousal', 'valence']])
							eval_Y = pd.DataFrame({'arousal': np.where(eval_Y['arousal'] >= 0, 1, 0), 'valence': np.where(eval_Y['valence'] >= 0, 1, 0)})
							
							
							
							y = [f'{Y['arousal'][a]}_{Y['valence'][a]}' for a in range(Y.shape[0])]
							for fold_idx, (train, test) in enumerate(splitter.split(X, y)):
								clf = MultiOutputClassifier(clone(clf_template))
								X_train, X_test = X.iloc[train], X.iloc[test]
								y_train, y_test = Y.iloc[train], Y.iloc[test]
								clf.fit(X_train, y_train)

								pred_y = clf.predict(X_test)
								eval_pred_y = clf.predict(eval_X)

								score = accuracy_score(y_test, pred_y)
								eval_score = accuracy_score(eval_Y, eval_pred_y)
								
								score_balanced_arousal = balanced_accuracy_score(y_test['arousal'], pred_y[:,0])
								eval_score_balanced_arousal = balanced_accuracy_score(eval_Y['arousal'], eval_pred_y[:,0])

								score_balanced_valence = balanced_accuracy_score(y_test['valence'], pred_y[:,1])
								eval_score_balanced_valence = balanced_accuracy_score(eval_Y['valence'], eval_pred_y[:,1])

								score_precision = precision_score(y_test, pred_y, average='macro')
								eval_score_precision = precision_score(eval_Y, eval_pred_y, average='macro')

								score_recall = recall_score(y_test, pred_y, average='macro')
								eval_score_recall = recall_score(eval_Y, eval_pred_y, average='macro')

								report = {
									'fold': fold_idx, 
									'classifier': classifier_id, 
									'splitter': splitter_id,
									'window': window, 
									'overlap': overlap, 
									'time_frame': time_frame,
									'repeat': repeat,
									'score': score, 
									'eval_score': eval_score,'score_balanced_arousal': score_balanced_arousal,
									'score_balanced_valence': score_balanced_valence,
									'eval_score_balanced_arousal': eval_score_balanced_arousal,
									'eval_score_balanced_valence': eval_score_balanced_valence,
									'score_precision': score_precision,
									'eval_score_precision': eval_score_precision,
									'score_recall': score_recall,
									'eval_score_recall': eval_score_recall,
									'type': 'features'
								}

								f_all_reports.append(report)
								
								print(f"Features: Fold {fold_idx} - Score: {score} - Eval Score: {eval_score}, Score Balanced: {(score_balanced_arousal + score_balanced_valence)/2}, Eval Score Balanced: {(eval_score_balanced_arousal + eval_score_balanced_valence)/2}, Score Recall: {score_recall}, Eval Score Recall: {eval_score_recall}")
						savepoint_keys[savepoint_key] = True
						save_checkpoint({
								"savepoint_keys": savepoint_keys,
								"f_all_reports": f_all_reports,
							}, CHECKPOINT_FILE)
	reports_df = pd.DataFrame(f_all_reports)
	print(reports_df)
	return f_all_reports

if( __name__ == '__main__'):
	#fix_chekpoints()
	reports = exp3()
	reports_df = pd.DataFrame(reports)
	reports_df.to_csv('exp3_features.csv', index=False)
	