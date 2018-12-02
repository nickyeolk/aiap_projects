import clustering

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

df_raw = pd.read_csv('data/raw/data.csv', encoding='ISO-8859-1')
df_clean = clustering.clean_data(df_raw,'./data/clean.pkl')

df_new = clustering.engineer_features1(df_clean,'./data/engineered_df.pkl')
X,df_inlier = clustering.transform_features(df_new)
model_comparison = clustering.compare_models(X,df_inlier)

# save the best model labels (KMEANS)
labeled = pd.DataFrame({'cluster':model_comparison[1]},index=df_inlier.index)
labeled.to_pickle('./data/labels.pkl')