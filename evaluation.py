from scripts.evaluate_multi import DGOT as DGOT_multi
from scripts.evaluate_binary import DGOT as DGOT_binary
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
import torch

torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)
np.random.seed(666)
os.environ['PYTHONHASHSEED'] = str(666)  # 为了禁止hash随机化，使得实验可复现。
#%%
_MODELS = {
	'binary_classification': [

		{
			'class': XGBClassifier,
			'kwargs': {
				'max_depth': 3,
				'n_estimators': 100
			}
		},

		{
			'class': DecisionTreeClassifier,
			'kwargs': {
				'max_depth': 30
			}
		},

		{
			'class': LogisticRegression,
			'kwargs': {
				'penalty': 'l2',
				'max_iter': 500
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

	methods = [ 'DGOT']
	columns  = ['accuracy', 'macro_f1','mcc']
	result = dict.fromkeys(methods, pd.DataFrame(columns=[x for x in columns]))

	filepaths_test = f'./datasets/{data_name}/TEST/exp{exp}'
	filepaths_DGOT = f'./saved_log/{model_name}/{data_name}/exp{exp}'
	repeat = args.repeats


	for i, classifier in enumerate(classifiers):

		model_param = classifier['kwargs']
		model = classifier['class'](**model_param)
		model_name = classifier['class'].__name__

		if args.task_name == 'binary':
			DGOT= DGOT_binary(filepaths_DGOT, filepaths_test, model, 1.2, repetitions=repeat)
		elif args.task_name == 'multi':
			DGOT= DGOT_multi(filepaths_DGOT, filepaths_test, model, 1.2, repetitions=repeat)

		result['DGOT'] = result['DGOT']._append(DGOT.head(repeat), ignore_index=True)


	result['DGOT'].to_csv(f'{result_filepath}/{data_name}.csv')





if __name__ == '__main__':
	parser = argparse.ArgumentParser('Test parameters')

	# data preparation
	parser.add_argument('--data_name', default='thyroid_sick', help='name of data')
	parser.add_argument('--task_name', default='binary', help='task of classification: binary/multi')

	# experiment param
	parser.add_argument('--exps', nargs='+',type=int, default=[0])
	parser.add_argument('--repeats', type=int, default=10)

	parser.add_argument('--model_name', default='DGOT', help='name of model')
	parser.add_argument('--result_file', default='./saved_log/evaluation', help='filepath of result')

	args = parser.parse_args()




	with tqdm(initial=0, total=len(args.exps)) as pbar:
		for exp in args.exps:

			data_name = args.data_name
			pbar.set_description(f"exp{exp}: {data_name} is being evaluated")
			args.data_name = data_name
			validation(args,exp)


			pbar.update(1)









