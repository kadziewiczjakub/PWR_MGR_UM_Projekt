import pickle
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from exp_utils import load_data_save_extracted_features

def prepare_data_for_exp1(prepared_data, train_games):
	train_data = [data[0] for data in prepared_data if data[0]['game'] in train_games]
	test_data = [data[0] for data in prepared_data if data[0]['game'] not in train_games]

	train_f = [data[1] for data in prepared_data if data[0]['game'] in train_games]
	test_f = [data[1] for data in prepared_data if data[0]['game'] not in train_games]
	return pd.DataFrame(train_data), pd.DataFrame(test_data), pd.concat(train_f), pd.concat(test_f)

def exp1():
	#windows = [2, 3, 6, 10, 20, 30, 60, 90, 120]
	windows = [2, 3, 10, 20, 60, 120] # TODO change if needed
	overlaps = [0, 0.25, 0.5, 0.75]
	game_ratios = [0.25, 0.5, 0.75] # 1 game, 2 games, 3 games in training sets
	N_REPEATS = 1 # TODO change
	SUBJECTS = list(range(1, 29))
	GAMES = list(range(1,5))
	CLASSIFIERS = [
		RandomForestClassifier(n_estimators=100, n_jobs=-1),
		RandomForestClassifier(n_estimators=500, n_jobs=-1),
		KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
		HistGradientBoostingClassifier(max_iter=1000),
		SVC()
	]

	SPLITTERS = [
		KFold(n_splits=5, shuffle=True),
		RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
		#StratifiedKFold(n_splits=5, shuffle=True),
	]
	all_reports = []
	for classifier_id, clf_template in enumerate(CLASSIFIERS):
		for splitter_id, splitter in enumerate(SPLITTERS):
			for window in windows:
				for overlap in overlaps:
					path = f'all_prepared_data_{window}s_overlap_{int(overlap*100)}.pkl'
					all_prepared_data = pickle.load(open(path, 'rb'))
					for repeat in range(N_REPEATS):
						for game_ratio in game_ratios:
							print(f"Classifier: {classifier_id}, Splitter: {splitter_id}, Window: {window}, Overlap: {overlap}, Game_ratio {game_ratio}, Repeat: {repeat}")

							nrs = GAMES.copy()
							np.random.shuffle(nrs)
							train_games = nrs[:int(len(nrs)*game_ratio)]
							_, _, train_f, test_f = prepare_data_for_exp1(all_prepared_data, train_games)

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
								
								report = {
									'fold': fold_idx, 
									'classifier': classifier_id, 
									'splitter': splitter_id, 
									'window': window, 
									'overlap': overlap, 
									'train_game_ratio': game_ratio,
									'repeat': repeat,
									'score': score, 
									'eval_score': eval_score
								}

								all_reports.append(report)
								
								print(f"Fold {fold_idx} - Score: {score} - Eval Score: {eval_score}")
	reports_df = pd.DataFrame(reports)
	print(reports_df)
	return reports

if __name__ == '__main__':
	reports = exp1()
	reports_df = pd.DataFrame(reports)
	reports_df.to_csv('exp1.csv', index=False)
	