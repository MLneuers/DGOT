from scripts.evaluate_multi import BTDG as BTDG_multi
from scripts.evaluate_binary import BTDG as BTDG_binary
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import json
from tqdm import tqdm

import argparse
import warnings

#%%
_MODELS = {
    'binary_classification': [
		{
            'class': XGBClassifier,
            'kwargs': {
				'n_estimators': 100
            }
		},
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
            }
        },

		{
			'class': LogisticRegression,
			'kwargs': {
				'penalty': 'l2',
				'max_iter':500
			}
		},

		{
            'class': RandomForestClassifier,
            'kwargs': {
				'n_estimators': 100
            }
        },


		{
            'class': KNeighborsClassifier,
            'kwargs': {
                'n_neighbors': 5
            }
        },

    ]
}
warnings.simplefilter(action='ignore', category=FutureWarning)

# utils
def exists(x):
    return x is not None

def means_std(dataframe):
	means = dataframe.mean(axis=0)
	std = dataframe.std(axis=0)
	dataframe = dataframe.append(means, ignore_index=True)
	dataframe = dataframe.append(std, ignore_index=True)

	return dataframe

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

#%%
def validation(args,exp):

	data_name = args.data_name
	model_name = args.model_name

	result_filepath = f'{args.result_file}/{model_name}/exp{exp}'##
	if not os.path.exists(result_filepath):
		os.makedirs(result_filepath)
	classifiers = _MODELS['binary_classification']

	methods = [ 'BTDG']
	columns  = ['accuracy', 'macro_f1','mcc']
	result = dict.fromkeys(methods, pd.DataFrame(columns=[x for x in columns]))



	filepaths_test = f'./datasets_prep/{data_name}/TEST/exp{exp}'
	filepaths_BTDG = f'./saved_log/{model_name}/{data_name}/exp{exp}'
	repeat = args.repeats

	logall=[]

	for i, classifier in enumerate(classifiers):
		if args.slog:
			slogs = args.slog[i]
			if slogs < 50:
				slogs = None
		else:
			slogs = None

		model_param = classifier['kwargs']
		model = classifier['class'](**model_param)
		model_name = classifier['class'].__name__

		if args.task_name == 'binary':
			BTDG, logs = BTDG_binary(filepaths_BTDG, filepaths_test, model, 1.2, slog=slogs, repetitions=repeat)
		elif args.task_name == 'multi':
			BTDG, logs = BTDG_multi(filepaths_BTDG, filepaths_test, model, 1.2, slog=slogs, repetitions=repeat)



		result['BTDG'] = result['BTDG']._append(BTDG.head(repeat), ignore_index=True)
		logall.append(logs)


	result['BTDG'].to_csv(f'{result_filepath}/{data_name}.csv')

	return logall




if __name__ == '__main__':
	parser = argparse.ArgumentParser('Test parameters')

	# data preparation
	parser.add_argument('--data_name', default='abalone_15', help='name of data')
	parser.add_argument('--task_name', default='binary', help='task of classification: binary/multi')

	# experiment param
	parser.add_argument('--exps', nargs='+',type=int, default=[0])
	parser.add_argument('--slog', nargs='+',type=int, default=None)
	parser.add_argument('--repeats', type=int, default=10)

	parser.add_argument('--model_name', default='DGOT', help='name of model')
	parser.add_argument('--result_file', default='./saved_log/evaluation', help='filepath of result')

	args = parser.parse_args()




	with tqdm(initial=0, total=len(args.exps)) as pbar:
		for exp in args.exps:
			lognumpy = []
			if os.path.exists(f'{args.result_file}/{args.model_name}/exp{exp}/logs_{args.data_name}.json'):
				with open(f'{args.result_file}/{args.model_name}/exp{exp}/logs_{args.data_name}.json', 'r') as json_file:
					data_logs = json.load(json_file)
			else:
				data_logs = None

			data_name = args.data_name

			pbar.set_description(f"exp{exp}: {data_name} is being evaluated")

			args.data_name = data_name
			if data_logs:
				args.slog = data_logs[data_name]
			else:
				args.slog = None

			temp = validation(args,exp)
			lognumpy.append({data_name:temp})

			json_temp = {data_name:temp}
			with open(f'{args.result_file}/{args.model_name}/exp{exp}/logs_{args.data_name}.json', 'w') as f:
				f.write(
					json.dumps(json_temp, ensure_ascii=False, indent=4, separators=(',', ':'), default=default_dump))


			pbar.update(1)









