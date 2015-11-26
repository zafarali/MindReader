# Contains Input-Output Utilities
# for handling data
import numpy as np
import pandas as pd
import os
import re


RAW_CHANNEL_NAMES = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
LABEL_NAMES = ['HandStart','FirstDigitTouch','BothStartLoadPhase','LiftOff','Replace','BothReleased']



def get_datadir(mode='train'):
	"""returns the location of the datadir"""
	module_dir = os.path.dirname( os.path.abspath(__file__) )
	data_dir = module_dir + '/data'
	data_dir += '/train' if mode=='train' else '/test'
	data_dir += '/'
	
	return data_dir


def get_file_list(mode='train', ftype='csv', fullpath=False):
	"""Returns the list of files"""
	datadir = get_datadir(mode=mode)
	flist = os.listdir(datadir)
	csvregex = re.compile('.*_data\.' + ftype)
	matchlist = map(csvregex.search, flist)

	if fullpath:
		csvlist = [datadir + x.group() for x in matchlist if x is not None]
	else:
		csvlist = [x.group() for x in matchlist if x is not None]

	return csvlist
    



def load_raw_train_data(subject_id=1,series_id=1):
	"""
		Loads a set of data from the training folder
		returns the training data, training labels and 
		the (RAW) feature names.
		@params:
			subject_id: the ID of the participant
			series_id: the ID of the recording session
		@returns:
			data: n * m numpy array of time ordered data
			events: n * k labels for each data point 
	"""
	file_name = 'data/train/subj'+str(subject_id)+'_series'+str(series_id)
	data = pd.read_csv(file_name+'_data.csv')
	events = pd.read_csv(file_name+'_events.csv')
	return data.values[:,1:].astype(float), events.values[:,1:].tolist()
	
	
def load_raw_test_data(subject_id=1,series_id=9):
	"""
		Loads a set of data from the testing folder
		returns the testing data only
		@params:
			subject_id: the ID of the participant
			series_id: the ID of the recording session
		@returns:
			data: n * m numpy array of time ordered data
	"""
	file_name = './data/test/subj'+str(subject_id)+'_series'+str(series_id)
	data = pd.read_csv(file_name+'_data.csv')
	return data.values[:,1:].astype(float)


def data_streamer(mode='train', num_sets='all', num_patients=12, num_series=8):
	"""
		Generator that streams data according to how it is best necessary
		Use as follows:
			ds = data_streamer()
			data = ds.next() 
			# do something with data
			data = ds.next()
			# do something with data
			etc etc.
	"""
	participants = xrange(1,num_patients+1) # 12 subjects
	train_series = xrange(1,num_series+1) # 8 series in train
	if mode == 'test' and num_series > 2:
		num_series = 2
	test_series = xrange(9,9+num_series) # 2 series in test
	loaded = 0
	if mode == 'train':
		for participant in participants:
			for series in train_series:
				yield load_raw_train_data(subject_id=participant, series_id=series)
				loaded += 1
				if num_sets != 'all' and num_sets < loaded: break
			if num_sets != 'all' and num_sets < loaded: break

	elif mode == 'test':
		for participant in participants:
			for series in test_series:
				yield load_raw_test_data(subject_id=participant, series_id=series)
				loaded += 1 
				if num_sets != 'all' and num_sets < loaded: break
			if num_sets != 'all' and num_sets < loaded: break
			



if __name__ == '__main__':
	print """
		To use IOUtils:
		(1) Download data from https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data
		(2) Copy 'test.zip' and 'train.zip' into a folder called data in this repository
		(3) Unzip the two.
		(4) good to go
	"""


