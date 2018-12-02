import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabaz_score#, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def remove_accounting(df):
    print('accounting entries',len(df[df['InvoiceNo'].str.contains('^[A]')]))
    return df[~df['InvoiceNo'].str.contains('^[A]')]

def remove_inventory_management(df):
    # Items with a negative quantity and InvoiceNo not starting with C are removed
    print('inventory adjustment items removed:',len(df[((df['Quantity']<0) & (df['InvoiceNo'].str.contains('^(?!C)')))]))
    return df[~((df['Quantity']<0) & (df['InvoiceNo'].str.contains('^(?!C)')))]

def remove_neg_price(df):
    print('Price at or below 0 removed: ',len(df[df['UnitPrice']<=0]))
    return df[df['UnitPrice']>0]

def remove_ghost_cust(df):
    print('Customers with no ID: ',np.sum(df['CustomerID'].isnull()))
    df.dropna(axis=0,subset=['CustomerID'],inplace=True)
    return df

def remove_neg_quantities(df):
    print('Quantities at or below 0 removed: ',len(df[df['Quantity']<=0]))
    return df[df['Quantity']>0]

def remove_dupe(df):
    print('Duplicates',len(df[df.duplicated(keep='first')]))
    return df.drop_duplicates()

def clean_data(df,filename): # This function calls all the clearning steps
    print('initial len',len(df))
    df = remove_accounting(df)
    df = remove_inventory_management(df)
    df = remove_neg_price(df)
    df = remove_ghost_cust(df)
    df = remove_neg_quantities(df)
    df = remove_dupe(df)
    df.reset_index(inplace=True,drop=True)
    print('final len',len(df))
    df.to_pickle(filename)
    print('clean data saved as ',filename)
    return df


## your code here 
def engineer_features1(df,filename):
    grouped = df.groupby('CustomerID')
    NoOfInvoices = grouped['InvoiceNo'].nunique()
    NoOfUniqueItems = grouped['StockCode'].nunique()
    QuantityPerInvoice = grouped['Quantity'].sum()/NoOfInvoices
    TotalQuantity = grouped['Quantity'].sum()
    UniqueItemsPerInvoice = NoOfUniqueItems/NoOfInvoices
    UnitPriceMean = grouped['UnitPrice'].mean()
    UnitPriceStd = grouped['UnitPrice'].std()
    UnitPriceStd.fillna(0,inplace=True)
    df_new = pd.DataFrame({'NoOfInvoices':NoOfInvoices,'NoOfUniqueItems':NoOfUniqueItems,'QuantityPerInvoice':QuantityPerInvoice,\
                     'TotalQuantity':TotalQuantity,'UniqueItemsPerInvoice':UniqueItemsPerInvoice,\
                     'UnitPriceMean':UnitPriceMean,'UnitPriceStd':UnitPriceStd})
    df_new.index = df_new.index.astype(int)
    df_new.to_pickle(filename)
    return df_new

def transform_features(df):
    '''Transforms features by removing the top 20th percentiles of each values and applying the StandardScaler'''
    df_inlier = df[(df['NoOfInvoices']<=df['NoOfInvoices'].quantile(0.8))& \
                (df['NoOfUniqueItems']<=df['NoOfUniqueItems'].quantile(0.8))& \
                (df['QuantityPerInvoice']<=df['QuantityPerInvoice'].quantile(0.8))& \
                (df['TotalQuantity']<=df['TotalQuantity'].quantile(0.8))& \
                (df['UniqueItemsPerInvoice']<=df['UniqueItemsPerInvoice'].quantile(0.8)) & \
                (df['UnitPriceMean']<=df['UnitPriceMean'].quantile(0.8)) & \
                (df['UnitPriceStd']<=df['UnitPriceStd'].quantile(0.8))]
    return StandardScaler().fit_transform(df_inlier),df_inlier


def compare_models(x_vals,df):
    nclusters = 3
    #define models
    models = [AgglomerativeClustering(n_clusters=nclusters, affinity='euclidean', linkage='ward') ,\
            KMeans(n_clusters=nclusters, init='k-means++', n_init=10, max_iter=300, \
                         tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, \
                         copy_x=True, n_jobs=None, algorithm='auto'), \
            GaussianMixture(n_components=nclusters, covariance_type='full') ]

    def make_figure_lowd(X,labels,fig,axii=[0,1,2]):
        X_low = TSNE(n_components=2,random_state=42).fit_transform(X)
        ax = fig.add_subplot(121)
        ax.scatter(X_low[:,0], X_low[:,1], c=labels, cmap='rainbow',alpha=0.7)
        ax.set_title('TSNE Breakdown')
    #     plt.colorbar()
        X_low2 = PCA(n_components=2).fit_transform(X)
        ax2 = fig.add_subplot(122)
        ax2.scatter(X_low2[:,0], X_low2[:,1], c=labels, cmap='rainbow',alpha=0.7)
        ax2.set_title('PCA Breakdown')
    #     plt.colorbar()

    def make_kde(df,fig_ct):
        plt.figure(fig_ct,figsize=(10, 7))
        columnss = df.columns
        for jj, coll in enumerate(columnss[:-1]):
        
            plt.subplot(2,4,jj+1)
            sns.distplot( df[df['labels']==0][coll] , color="skyblue", label='0',hist=False)
            sns.distplot( df[df['labels']==1][coll] , color="red", label='1',hist=False)
            sns.distplot( df[df['labels']==2][coll] , color="green", label='2',hist=False)
        plt.show()
        fig_ct += 1
        return fig_ct


    label_results=[]
    fig_ct=1
    for ii,model in enumerate(models):
        if ii<2:
            model.fit_predict(x_vals)
            labels = model.labels_
        else: #GMM model is special
            model.fit(x_vals)
            labels = model.predict(x_vals)
        #calculate objective scores
        silh_score = silhouette_score(x_vals, labels, metric='euclidean')
        calinsky_score = calinski_harabaz_score(x_vals, labels)
        print('Silhouette score: {:.4f}, Calinsky harabaz score: {:.4f}'.format(silh_score,calinsky_score))
        # Make lower dimension plots
        fig = plt.figure(fig_ct,figsize=(10, 7)) 
        make_figure_lowd(x_vals,labels,fig)
        plt.show()
        fig_ct+=1
        # Inspect
        df['labels'] = labels
        print(df.groupby('labels').mean())
        make_kde(df,fig_ct)
        label_results.append(labels)
    return label_results
    