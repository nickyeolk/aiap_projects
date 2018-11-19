# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:08:05 2018

@author: likkhian
"""
import numpy as np
class DecisionTree():
    def __init__(self,max_depth=5,min_samples_split=1,debug=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.debug = debug
    #def gini(self, a1, a2):
    def gini(self,a1,a2):
        def ginicalc(array):
            positives = sum(array)
            total = len(array)
            return 1 - (positives/total)**2 - (1-positives/total)**2
        w1 = len(a1)/(len(a1)+len(a2))
        w2 = 1-w1
        return ginicalc(a1)*w1 + ginicalc(a2)*w2
    
    
    def fit(self,x,y):
        def split_data(feature,cutoff,xx,yy):
            mask = xx[:,feature] < cutoff
            left_y = yy[mask]
            left_x = xx[mask]
            right_y = yy[~mask]
            right_x = xx[~mask]
            return left_x,left_y,right_x,right_y
            
        def get_split(x,y):
            nsamp,nfeat = np.shape(x)
            if self.debug:
                print('x shape',np.shape(x),np.shape(y))
            min_gini=999 # 0.5
            for feature in range(nfeat):
                unique_vals = np.unique(x[:,feature])
                if self.debug:
                    print('unique_vals',unique_vals)
                for ii in range(1,len(unique_vals)):
                    left_x, left_y, right_x, right_y = split_data(feature,unique_vals[ii],x,y)
                    gini_val = self.gini(left_y,right_y)
                    if self.debug:
                        print('gini', gini_val)
                    if gini_val<min_gini:
                        min_gini=gini_val
                        location=[feature,unique_vals[ii]]
                        best_left_x,best_left_y,best_right_x,best_right_y = left_x, left_y, right_x, right_y
#            print('split deet',location,min_gini)
            return {'index':location[0],'value':location[1], \
                'groups': [best_left_x, best_left_y, best_right_x, best_right_y]}
            
    #    get_split(x,y)
        
        def get_consensus(leaf):
            # note: if leafs are equal then peeps gon die.
            return np.bincount(leaf).argmax()
            
        def check_pure_feat(x_array):
            row,col = np.shape(x_array)
            for ii in range(col):
                if len(np.unique(x_array[:,ii])) > 2:
                    return False
            return True
        
        def split(node,depth):
            left_x, left_y, right_x, right_y = node['groups']
#            print('shaply',np.shape(left_x),np.shape(left_y),np.shape(right_x),np.shape(right_y))
            del(node['groups'])
#            print(left_y,right_y,'sizes')
            # check if either group is empty
            if((not left_y.size) or (not right_y.size) ):
                node['left'] = node['right'] = get_consensus(list(left_y)+list(right_y))
#                print('grp empty')
                return
            # check if we be deep
            if depth >= self.max_depth:
                node['left'], node['right'] = get_consensus(left_y),get_consensus(right_y)
                if self.debug:
                    print('too deep!')
                return
            # check if we are at the left end
            if len(left_y) <= self.min_samples_split:
                node['left'] = get_consensus(left_y)
            elif len(np.unique(left_y)) == 1: #leaf is pure
                node['left'] = get_consensus(left_y)
            elif check_pure_feat(left_x): #features are pure
                node['left'] = get_consensus(left_y)
            else:
                node['left'] = get_split(left_x,left_y)
                split(node['left'],depth+1)
            # do the right thing
            if len(right_y) <= self.min_samples_split:
                node['right'] = get_consensus(right_y)
            elif len(np.unique(right_y)) == 1: #leaf is pure
                node['right'] = get_consensus(right_y)
            elif check_pure_feat(right_x): #features are pure
                node['right'] = get_consensus(right_y)
            else:
                node['right'] = get_split(right_x,right_y)
                split(node['right'],depth+1)
        
        def grow_tree(train_x,train_y):
            root = get_split(train_x,train_y)
            split(root,1)
            return root
        self.mature_tree = grow_tree(x,y)
#        return grow_tree(x,y)
        
    def konica(self,node, depth=0):
        if isinstance(node, dict):
            print('hi, %s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            konica(node['left'], depth+1)
            konica(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))
        
#x = np.array([[5,4,3,2,1,9,9,9,9,9],[9,9,9,9,9,1,2,3,4,5]]).T
#y = np.array([1,1,1,1,1,0,0,0,0,0])
##fit(x,y)
#
#dataset = np.array([[2.771244718,1.784783929],
#	[1.728571309,1.169761413],
#	[3.678319846,2.81281357],
#	[3.961043357,2.61995032],
#	[2.999208922,2.209014212],
#	[7.497545867,3.162953546],
#	[9.00220326,3.339047188],
#	[7.444542326,0.476683375],
#	[10.12493903,3.234550982],
#	[6.642287351,3.319983761]])
#dataset = np.array([[5,8],
#	[5,8],
#	[5,8],
#	[5,1],
#	[5,1],
#	[2,8],
#	[2,8],
#	[2,8],
#	[2,1],
#	[2,1]])
#y = np.array([0,0,0,1,1,1,1,1,0,0])
#tree = fit(dataset,y,max_depth=3,min_samples_split=3)
#konica(tree)

    
    def predict(tree,x):
        prediction = []
        def predictor(tree,row):
            if row[tree['index']] < tree['value']:
                if isinstance(tree['left'],dict):
                    return predictor(tree['left'],row)
                else: return tree['left']
            else:
                if isinstance(tree['right'],dict):
                    return predictor(tree['right'],row)
                else: return (tree['right'])        
        
        for row in x:
            prediction.append(predictor(tree.mature_tree,row))
        return np.array(prediction)

#predict(tree,dataset)
class RandomForest():
    def __init__(self,num_trees,max_depth=5,subsample_size=1.0,feature_proportion=1.0):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.feature_proportion = feature_proportion
#        print('random forested')
        
    def fit(self,x,y):
        self.forest = []
#        print('random forest 0')
        for run in range(self.num_trees):
#            print('random forest stage ',run)
            sample_index = np.random.choice(len(y),int(self.subsample_size*(len(y))))
            x_sample = x[sample_index,:]
            y_sample = y[sample_index]
            feature_index = np.random.choice(np.shape(x_sample)[1], \
            int(self.feature_proportion*np.shape(x_sample)[1]),replace=False)
            x_sample = x_sample[:,feature_index]
            current_tree = DecisionTree(max_depth = self.max_depth)
            current_tree.fit(x_sample,y_sample)
            self.forest.append(current_tree)
    
    def predict(self,x):
        results = []
        for tree in self.forest:
            results.append(tree.predict(x))
            mean_results = np.mean(results,axis=0)
            mean_results[mean_results>0.5] = 1
            mean_results[mean_results<=0.5] = 0
            return mean_results
            
            
        
        
            
            
            