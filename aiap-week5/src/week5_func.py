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

def read_to_df(dirname):
    scores = []
    text = []
    for filename in os.listdir(dirname):
        scores.append(re.search('\_(.*?)\.',filename).group(1))
        file = open(dirname+filename,'r',encoding='utf-8')
        try:
            text.append(file.read())
        except:
            print(filename)
    scores = [int(i) for i in scores]
    return pd.DataFrame({'text':text, 'scores':scores})

def save_as_pkl():
	df_pos = read_files('C:/Users/likkhian/Desktop/aclImdb/train/pos/')
	df_pos['positive'] = 1
	df_neg = read_files('C:/Users/likkhian/Desktop/aclImdb/train/neg/')
	df_neg['positive'] = 0
	df_raw = pd.concat([df_pos,df_neg])

	df_pos_test = read_files('C:/Users/likkhian/Desktop/aclImdb/test/pos/')
	df_pos_test['positive'] = 1
	df_neg_test = read_files('C:/Users/likkhian/Desktop/aclImdb/test/neg/')
	df_neg_test['positive'] = 0
	df_raw_test = pd.concat([df_pos_test,df_neg_test])
	df_raw.to_pickle('./data/df_raw.pkl')
	df_raw_test.to_pickle('./data/df_raw_test.pkl')

def just_dataframes(filepath):
    '''This function can be used to quickly load the
    data of a single training set as a dataframe. 
    '''
    import pandas as pd
    dictionary = unpickle(filepath)
    df = pd.DataFrame(dictionary[b'data'])
    df['target'] = dictionary[b'labels']
    return df
