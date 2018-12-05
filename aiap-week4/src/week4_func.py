# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:36:21 2018
Functions to help with week4 assignments
@author: likkhian
"""

#import requests
import os.path

def untar(filepath):
    print(filepath)
    import tarfile
    tar = tarfile.open(filepath,'r:gz')
    tar.extractall('./data/')

def dl_and_unzip(url):
    import wget
    filename = url.split('/')[-1]
    if os.path.isfile('./data/'+filename):
        print(filename,' already exists' )
    else:
        print('downloading ',filename)
        wget.download(url,out='./data/')
#        with open('./data/'+filename, 'wb') as f:
#            req = requests.get(url,stream = True)
#            for chunk in req.iter_content(chunk_size=1024):
#                f.write(req.content)
        print('Done!')
    untar('./data/'+filename)
    print('extracted!')

def unpickle(filepath):
    import pickle
    with open(filepath, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def just_dataframes(filepath):
    '''This function can be used to quickly load the
    data of a single training set as a dataframe. 
    '''
    import pandas as pd
    dictionary = unpickle(filepath)
    df = pd.DataFrame(dictionary[b'data'])
    df['target'] = dictionary[b'labels']
    return df
