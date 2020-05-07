import operator
import numpy as np
from numpy import *
from collections import Counter

'''
    train_data_set
    ID	AGE	    JOB	HOUSE	CREDIT	    GIVE LOANS
    1	youth	no	no	    general	        no
    2	youth	no	no	    good	        no
    3	youth	yes	no	    good	        yes
    4	youth	yes	yes	    general	        yes
    5	youth	no	no	    general	        no
    6	middle	no	no	    general	        no
    7	middle	no	no	    good	        no
    8	middle	yes	yes	    good	        yes
    9	middle	no	yes	    very good	    yes
    10	middle	no	yes	    very good	    yes
    11	old	    no	yes	    very good	    yes
    12	old	    no	yes	    good	        yes
    13	old	    yes	no	    good	        yes
    14	old	    yes	no	    very good	    yes
    15	old	    no	no	    general	        no
    16	old	    no	no	    very good	    no
'''

class CART(object):

    def __init__(self):
        data_set = [
            ['youth', 'no', 'no', 'general', 'deny'],
            ['youth', 'no', 'no', 'good', 'deny'],
            ['youth', 'yes', 'no', 'good', 'approve'],
            ['youth', 'yes', 'yes', 'general', 'approve'],
            ['youth', 'no', 'no', 'general', 'deny'],
            ['middle', 'no', 'no', 'general', 'deny'],
            ['middle', 'no', 'no', 'good', 'deny'],
            ['middle', 'yes', 'yes', 'good', 'approve'],
            ['middle', 'no', 'yes', 'very good', 'approve'],
            ['middle', 'no', 'yes', 'very good', 'approve'],
            ['old', 'no', 'yes', 'very good', 'approve'],
            ['old', 'no', 'yes', 'good', 'approve'],
            ['old', 'yes', 'no', 'good', 'approve'],
            ['old', 'yes', 'no', 'very good', 'approve'],
            ['old', 'no', 'no', 'general', 'deny'],
            ['old', 'no', 'no', 'very good', 'deny'],
                    ]

        lables = ["AGE","JOB","HOUSE","CREDIT"]
        self.data_set = data_set
        self.lables = lables


    #划分数据集，提取含有某个特征的某个属性的所有数据
    def splitDataset(self,data_set, index, value):
        left_dataset = []
        right_dataset = []
        for feat_vec in data_set :
            if feat_vec[index] == value:
               left_dataset.append(feat_vec[:index] + feat_vec[index+1:])
            else:
               right_dataset.append( feat_vec[:index] + feat_vec[index+1:])
        return left_dataset,right_dataset

    # 返回一个dataset的基尼系数
    def gini(self,data_set):
        gn = {
            'no':0,
            'yes':1,
            'deny':0,
            'approve':1,
            'youth':0,
            'middle':1,
            'old':2,
            'general':0,
            'good':1,
            'very good':2
        }
        gini = 0
        rows = len(data_set)
        features = len(data_set[0]) - 1
        array = mat(data_set)
        dic = Counter(np.transpose(array[:, features]).tolist()[0])
        for item in dic:
            gini += (gn[item] / rows) ** 2
        return 1 - gini

    # 根据基尼系数选择当前数据集的最优划分特征
    def chooseBestFeature(self,data_set):
        features = len(data_set[0]) - 1
        if features == 1:
            return 0
        rows = len(data_set)
        best_gini = 0
        best_feature = -1
        array = mat(data_set)
        giniloan = self.gini(data_set)
        for i in range(features):
            count = Counter(np.transpose(array[:, i]).tolist()[0])
            for item in count:
                l,r = self.splitDataset(data_set,i, item)
                l_gini = self.gini(l)
                r_gini = self.gini(r)
                gini = giniloan - int(count[item]) / rows * l_gini - (
                            rows - int(count[item])) / rows * r_gini
                if gini > best_gini:
                    best_gini = gini
                    best_feature = i
        return best_feature

    def createTree(self,data_set,lables):
        classList = [each[-1] for each in data_set]
        if classList.count(classList[0]) == len(classList):
            return classList[0]

        best_feat = self.chooseBestFeature(data_set)
        best_feat_label = lables[best_feat]
        CART_tree = {best_feat_label: {}}
        del (lables[best_feat])
        feat_values = [each[best_feat] for each in data_set]
        unique_vls = set(feat_values)
        for value in unique_vls:
            l,r = self.splitDataset(data_set, best_feat, value)
            temp_lables = lables[:]
            CART_tree[best_feat_label][value] = self.createTree(l,temp_lables)
        return CART_tree

    def classify(self,tree, test):
        firstStr = next(iter(tree))
        secondDict = tree[firstStr]
        featIndex = self.lables.index(firstStr)
        for key in secondDict.keys():
            if test[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel =self.classify(secondDict[key], test)
                else:
                    classLabel = secondDict[key]
        return classLabel


    def search(self,test):
        lables = self.lables[:]
        tree = self.createTree(self.data_set,lables)
        return self.classify(tree,test)







if __name__ == '__main__':
    cart = CART()
    print( cart.search(['old', 'no', 'no', 'general']))


