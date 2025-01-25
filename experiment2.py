import pickle
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from exp_utils import load_data_save_extracted_features
import os




def save_checkpoint(state, CHECKPOINT_FILE):
	with open(CHECKPOINT_FILE, "wb") as f:
		pickle.dump(state, f)


def load_checkpoint(CHECKPOINT_FILE):
	if os.path.exists(CHECKPOINT_FILE):
		with open(CHECKPOINT_FILE, "rb") as f:
			return pickle.load(f)
	return None


def prepare_data_for_exp2(prepared_data, train_subjects):
	train_data = [data[0] for data in prepared_data if data[0]['subject'] in train_subjects]
	test_data = [data[0] for data in prepared_data if data[0]['subject'] not in train_subjects]


	train_X = [data['data'] for data in train_data]
	train_Y = [(data['arousal'], data['valence']) for data in train_data]
	
	tr_x = []
	tr_y = []
	#transform data into df
	for i in range(len(train_X)):
		tmp_y = {'arousal': 1 if train_Y[i][0] >= 0 else 0, 'valence': 1 if train_Y[i][1] >= 0 else 0}
		for j in range(len(train_X[i])):
			mapp = {}
			for k in range(len(train_X[i][j])):
				mapp[f'ch{k}'] = train_X[i][j][k]
			tr_x.append(mapp)
			tr_y.append(tmp_y)

	scaler = MinMaxScaler()

	tr_x = pd.DataFrame(tr_x)
	tr_x = pd.DataFrame({f'{col}_{i}': tr_x[col].apply(lambda x: x[i]) for col in tr_x.columns for i in range(len(tr_x[col].iloc[0]))}) # Flatten
	tr_x = pd.DataFrame(scaler.fit_transform(tr_x))
	tr_y = pd.DataFrame(tr_y)
	train_data = [tr_x, tr_y]

	test_X = [data['data'] for data in test_data]
	test_Y = [(data['arousal'], data['valence']) for data in test_data]
	te_x = []
	te_y = []
	for i in range(len(test_X)):
		tmp_y = {'arousal': 1 if test_Y[i][0] >= 0 else 0, 'valence': 1 if test_Y[i][1] >= 0 else 0}
		for j in range(len(test_X[i])):
			mapp = {}
			for k in range(len(test_X[i][j])):
				mapp[f'ch{k}'] = test_X[i][j][k]
			te_x.append(mapp)
			te_y.append(tmp_y)

	te_x = pd.DataFrame(te_x)
	te_x = pd.DataFrame({f'{col}_{i}': te_x[col].apply(lambda x: x[i]) for col in te_x.columns for i in range(len(te_x[col].iloc[0]))})# Flatten
	te_x = pd.DataFrame(scaler.transform(te_x))
	te_y = pd.DataFrame(te_y)
	test_data = [te_x, te_y]

	train_f = [data[1] for data in prepared_data if data[0]['subject'] in train_subjects]
	test_f = [data[1] for data in prepared_data if data[0]['subject'] not in train_subjects]
	return train_data, test_data, pd.concat(train_f), pd.concat(test_f)


def exp2():
	# Checkpoint file
	CHECKPOINT_FILE = "exp2_checkpoint.pkl"
	windows = [3, 6, 10, 20, 30, 60, 120]
	overlaps = [0, 0.25, 0.5, 0.75]
	subject_ratios = [0.25, 0.5, 0.75]
	N_REPEATS = 1
	SUBJECTS = list(range(1, 29))
	CLASSIFIERS = [
		RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=99),
		RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=99),
		KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
		SVC(random_state=99),
		#None,#HistGradientBoostingClassifier(max_iter=100),
		
	]

	SPLITTERS = [
		RepeatedKFold(n_splits=5, n_repeats=2, random_state=99),
		RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=99),
	]

	# Load checkpoint if available
	checkpoint = load_checkpoint(CHECKPOINT_FILE)
	if checkpoint:
		r_all_reports = checkpoint["r_all_reports"] #raw reports
		f_all_reports = checkpoint["f_all_reports"]	#features reports
		savepoint_keys = checkpoint["savepoint_keys"]
		print("Resuming from checkpoint...")
	else:
		f_all_reports = []
		r_all_reports = []
		savepoint_keys = {}
	for repeat in range(N_REPEATS):
		for overlap in overlaps:
			for splitter_id, splitter in enumerate(SPLITTERS):
				for classifier_id, clf_template in enumerate(CLASSIFIERS):
					for window in windows:
						savepoint_key = f'{repeat}_{classifier_id}_{splitter_id}_{window}_{overlap}'
						if savepoint_key in savepoint_keys:
							continue
						path = f'all_prepared_data_{window}s_overlap_{int(overlap*100)}.pkl'
						all_prepared_data = pickle.load(open(path, 'rb'))

						for subject_ratio in subject_ratios:
							print(f"Classifier: {classifier_id}, Splitter: {splitter_id}, Window: {window}, Overlap: {overlap}, Subject_ratio {subject_ratio}, Repeat: {repeat}")

							nrs = SUBJECTS.copy()
							np.random.seed(99+repeat)
							np.random.shuffle(nrs)
							train_subjects = nrs[:int(len(nrs)*subject_ratio)]
							train_r, test_r, train_f, test_f = prepare_data_for_exp2(all_prepared_data, train_subjects)

							X_f = train_f.drop(columns=['arousal', 'valence', 'subject', 'game'], errors='ignore')
							Y_f = pd.DataFrame(train_f[['arousal', 'valence']])
							Y_f = pd.DataFrame({'arousal': np.where(Y_f['arousal'] >= 0, 1, 0), 'valence': np.where(Y_f['valence'] >= 0, 1, 0)})

							eval_X_f = test_f.drop(columns=['arousal', 'valence', 'subject', 'game'], errors='ignore')
							eval_Y_f = pd.DataFrame(test_f[['arousal', 'valence']])
							eval_Y_f = pd.DataFrame({'arousal': np.where(eval_Y_f['arousal'] >= 0, 1, 0), 'valence': np.where(eval_Y_f['valence'] >= 0, 1, 0)})

							y = [f'{Y_f['arousal'][a]}_{Y_f['valence'][a]}' for a in range(Y_f.shape[0])]
							for fold_idx, (train, test) in enumerate(splitter.split(X_f, y)):
								clf = MultiOutputClassifier(clone(clf_template))
								X_train, X_test = X_f.iloc[train], X_f.iloc[test]
								y_train, y_test = Y_f.iloc[train], Y_f.iloc[test]
								clf.fit(X_train, y_train)

								pred_y = clf.predict(X_test)
								eval_pred_y = clf.predict(eval_X_f)

								score = accuracy_score(y_test, pred_y)
								eval_score = accuracy_score(eval_Y_f, eval_pred_y)
								
								score_balanced_arousal = balanced_accuracy_score(y_test['arousal'], pred_y[:,0])
								eval_score_balanced_arousal = balanced_accuracy_score(eval_Y_f['arousal'], eval_pred_y[:,0])

								score_balanced_valence = balanced_accuracy_score(y_test['valence'], pred_y[:,1])
								eval_score_balanced_valence = balanced_accuracy_score(eval_Y_f['valence'], eval_pred_y[:,1])

								score_precision = precision_score(y_test, pred_y, average='macro')
								eval_score_precision = precision_score(eval_Y_f, eval_pred_y, average='macro')

								score_recall = recall_score(y_test, pred_y, average='macro')
								eval_score_recall = recall_score(eval_Y_f, eval_pred_y, average='macro')

								report = {
									'fold': fold_idx, 
									'classifier': classifier_id, 
									'splitter': splitter_id, 
									'window': window, 
									'overlap': overlap, 
									'train_subject_ratio': subject_ratio,
									'repeat': repeat,
									'score': score, 
									'eval_score': eval_score,
									'score_balanced_arousal': score_balanced_arousal,
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
								
								print(f"Features: Fold {fold_idx} - Score: {score} - Eval Score: {eval_score}, Score Balanced: {(score_balanced_arousal + score_balanced_valence)/2}, Eval Score Balanced: {(eval_score_balanced_arousal + eval_score_balanced_valence)/2},\r\nScore Precision {score_precision}; {eval_score_precision} , Score Recall: {score_recall}; {eval_score_recall}")
							
							X_r, Y_r = train_r
							eval_X_r, eval_Y_r = test_r
							y = [f'{Y_r['arousal'][a]}_{Y_r['valence'][a]}' for a in range(Y_r.shape[0])]
							for fold_idx, (train, test) in enumerate(splitter.split(X_r, y)):
								clf = MultiOutputClassifier(clone(clf_template))
								X_train, X_test = X_r.iloc[train], X_r.iloc[test]
								y_train, y_test = Y_r.iloc[train], Y_r.iloc[test]
								clf.fit(X_train, y_train)

								pred_y = clf.predict(X_test)
								eval_pred_y = clf.predict(eval_X_r)

								score = accuracy_score(y_test, pred_y)
								eval_score = accuracy_score(eval_Y_r, eval_pred_y)

								score_balanced_arousal = balanced_accuracy_score(y_test['arousal'], pred_y[:,0])
								eval_score_balanced_arousal = balanced_accuracy_score(eval_Y_r['arousal'], eval_pred_y[:,0])

								score_balanced_valence = balanced_accuracy_score(y_test['valence'], pred_y[:,1])
								eval_score_balanced_valence = balanced_accuracy_score(eval_Y_r['valence'], eval_pred_y[:,1])

								score_precision = precision_score(y_test, pred_y, average='macro')
								eval_score_precision = precision_score(eval_Y_r, eval_pred_y, average='macro')

								score_recall = recall_score(y_test, pred_y, average='macro')
								eval_score_recall = recall_score(eval_Y_r, eval_pred_y, average='macro')
								
								report = {
									'fold': fold_idx, 
									'classifier': classifier_id, 
									'splitter': splitter_id, 
									'window': window, 
									'overlap': overlap, 
									'train_subject_ratio': subject_ratio,
									'repeat': repeat,
									'score': score, 
									'eval_score': eval_score,
									'score_balanced_arousal': score_balanced_arousal,
									'score_balanced_valence': score_balanced_valence,
									'eval_score_balanced_arousal': eval_score_balanced_arousal,
									'eval_score_balanced_valence': eval_score_balanced_valence,
									'score_precision': score_precision,
									'eval_score_precision': eval_score_precision,
									'score_recall': score_recall,
									'eval_score_recall': eval_score_recall,
									'type': 'raw'
								}

								r_all_reports.append(report)
								
								print(f"Raw: Fold {fold_idx} - Score: {score} - Eval Score: {eval_score}, Score Balanced: {(score_balanced_arousal + score_balanced_valence)/2}, Eval Score Balanced: {(eval_score_balanced_arousal + eval_score_balanced_valence)/2},\r\nScore Precision {score_precision}; {eval_score_precision} , Score Recall: {score_recall}; {eval_score_recall}")
						savepoint_keys[savepoint_key] = True
						save_checkpoint({
								"savepoint_keys": savepoint_keys,
								"f_all_reports": f_all_reports,
								"r_all_reports": r_all_reports
							}, CHECKPOINT_FILE)

	f_reports_df = pd.DataFrame(f_all_reports)
	r_reports_df = pd.DataFrame(r_all_reports)
	print(f_reports_df)
	print(r_reports_df)

	return f_all_reports, r_all_reports


if __name__ == '__main__':
	reports_f, reports_r = exp2()
	f_reports_df = pd.DataFrame(reports_f)
	r_reports_df = pd.DataFrame(reports_r)
	f_reports_df.to_csv('exp2_features.csv', index=False)
	r_reports_df.to_csv('exp2_raw.csv', index=False)
