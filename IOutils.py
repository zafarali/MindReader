# Contains Input-Output Utilities
# for handling data
from collections import Counter
import numpy as np
import pandas as pd
import os
import re


RAW_CHANNEL_NAMES = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']
LABEL_NAMES = ['HandStart','FirstDigitTouch','BothStartLoadPhase','LiftOff','Replace','BothReleased']



#------------------------------
def get_datadir(mode='train'):
    """returns the location of the datadir"""
    module_dir = os.path.dirname( os.path.abspath(__file__) )
    data_dir = module_dir + '/data'
    data_dir += '/train' if mode=='train' else '/test'
    data_dir += '/'
    
    return data_dir


#------------------------------
def get_file_list(mode='train', fullpath=False, regex='.*data\.csv'):
    """Returns the list of files"""
    datadir = get_datadir(mode=mode)
    flist = os.listdir(datadir)
    csvregex = re.compile(regex)
    matchlist = map(csvregex.search, flist)

    if fullpath:
        csvlist = [datadir + x.group() for x in matchlist if x is not None]
    else:
        csvlist = [x.group() for x in matchlist if x is not None]

    return csvlist
    

#------------------------------
def data_streamer2(mode='train', regex='.*W256_normFULL\.npy', keeplist=[], invertkeep=False):
    """Streams files from the data folder matching the regex.
    @params:
    keeplist: list of tuples of (subject, series) to keep. If empty, keep all.
              if (subject, 0), it will keep all the series for that subject
    invertkeep: If true, keeplist will behave as a list of (subject, series) to omit
    """
    pathlist = get_file_list(mode=mode, fullpath=True, regex=regex)
    flist = [os.path.split(x)[1] for x in pathlist]

    max_sub = 12
    max_series = 8 if mode=='train' else 2

    # Make a list of numbers corresponding to the subject_id and series_id
    matchlist = [re.finditer('\d+', fname) for fname in flist]
    idlist = [int(m.next().group(0))*100+int(m.next().group(0)) for m in matchlist]

    # Sort the data path list
    datadir = get_datadir(mode=mode)
    sorted_pathlist = sort_with_other(pathlist, idlist)
    idlist.sort()
    sorted_eventpathlist = [datadir + 'subj' + str(i/100) + '_series'\
                            + str(i%100) + '_events.csv' for i in idlist]

    # Expand keeplist
    exkeeplist = []
    for sub, series in keeplist:
        if series==0:
            exkeeplist += [max_series*(sub-1) + x - 1 for x in range(max_series)]
        else:
            exkeeplist.append(max_series*(sub-1) + series - 1)

    # SO that the default loads everything
    if exkeeplist==[]:
        invertkeep = True

    # Build omitlist
    if not invertkeep:
        omitlist = [x for x in range(len(sorted_pathlist)) if x not in exkeeplist]
    else:
        omitlist = exkeeplist

    
    # Located all omitted sub, series
    for curr_omit in omitlist:
        sorted_pathlist[curr_omit] = 0
        sorted_eventpathlist[curr_omit] = 0

    # Drop omitted sub, series
    sorted_pathlist = [x for x in sorted_pathlist if x is not 0]
    sorted_eventpathlist = [x for x in sorted_eventpathlist if x is not 0]
    
    
    #Iterator loop
    for path, event in zip(sorted_pathlist, sorted_eventpathlist):
        data = np.load(path)
        #data = path
        event_data = pd.read_csv(event).values[:,1:].astype(np.int32) if mode=='train' else None
        #event_data = event if mode=='train' else None
        yield data, event_data


#------------------------------
def sort_with_other(tosort,indexes):
    return [x for (x,y) in sorted(zip(tosort,indexes), key=lambda tmp: tmp[1])]

#------------------------------
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
    return data.values[:,1:].astype(float), events.values[:,1:].astype(np.int32)
    
    



#------------------------------
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


#------------------------------
def data_streamer(mode='train', num_sets='all', patients_list=None, series_list=None, num_patients=12, num_series=8):
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

    if patients_list:
        participants = patients_list
    else:
        participants = xrange(1,num_patients+1) # 12 subjects
    if series_list:
        train_series = series_list
        test_series = series_list
    else:
        test_series = xrange(9,9+num_series) # 2 series in test
        train_series = xrange(1,num_series+1) # 8 series in train

    if mode == 'test' and num_series > 2:
        num_series = 2
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
            
##### TRANSFORMER



# def _unique_rows(data):
#     uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
#     return uniq.view(data.dtype).reshape(-1, data.shape[1])


class VectorTransformer(object):
    def __init__(self, Y=None, prefit=True):
        """
            Train a transformer using an initial Y 
            that contains all possible vectors.
        """
        # if Y != None:
        #     raise FutureWarning('The usage of VectorTransformer has changed, please use as: \
        #         vt = VectorTransformer()\n\
        #         vt.fit(Y) \n\
        #         vt.transform(Y)')
        self.unique_rows = [(0, 0, 0, 0, 0, 0),
                            (0, 1, 1, 0, 0, 0),
                            (0, 1, 1, 1, 0, 0),
                            (1, 0, 0, 0, 0, 0),
                            (0, 0, 1, 0, 0, 0),
                            (0, 1, 0, 0, 0, 0),
                            (0, 0, 0, 0, 1, 1),
                            (0, 0, 1, 1, 0, 0),
                            (0, 0, 0, 1, 0, 0),
                            (0, 1, 0, 1, 0, 0),
                            (0, 0, 0, 0, 0, 1),
                            (0, 0, 0, 0, 1, 0)]
        self.silent = True
        self.prefit = prefit

    def fit(self, Y):
        """
            Refits the model "VectorTransformer" according
            to a new sample of Y
        """
        if prefit:
            return
        if type(Y) == np.ndarray:
            Y = map(lambda x: tuple(x), Y.tolist())

        self.unique_rows = Counter(Y).keys()
        self.silent = False


    def transform_vector(self, y):
        """
            Transforms a single vector y
        """
        y = tuple(y)
        for i, u in enumerate(self.unique_rows):
            if hash(y) == hash(u):
                return i
        if not self.silent:
            print y,' did not have a mapping'
        return -1

    def transform(self, Y):
        """
            Transforms an array of vectors according to the
            fitted model
        """
        return np.apply_along_axis(self.transform_vector, axis=1, arr=Y).astype(np.int32)



if __name__ == '__main__':
    olist = [(2,0),(1,3),(5,6)]
    tmp = data_streamer2(keeplist=olist)
    for dat, ev in tmp:
        print(dat.shape)
    
    print """
        To use IOUtils:
        (1) Download data from https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data
        (2) Copy 'test.zip' and 'train.zip' into a folder called data in this repository
        (3) Unzip the two.
        (4) good to go
    """


