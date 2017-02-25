"""
CS440/640: Lab-Week5
"""

import numpy as np 
import pandas



class LogisticRegression:
    
    def __init__(self, input_dim, output_dim):

        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):

        one_hot_y = [0]*len(y)
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        for i in range(len(y)):
            if y[i] == 0:
                one_hot_y[i] = [1,0,0,0,0,0,0,0,0,0]
            if y[i] == 1:
                one_hot_y[i] = [0,1,0,0,0,0,0,0,0,0]
            if y[i] == 2:
                one_hot_y[i] = [0,0,1,0,0,0,0,0,0,0]
            if y[i] == 3:
                one_hot_y[i] = [0,0,0,1,0,0,0,0,0,0]
            if y[i] == 4:
                one_hot_y[i] = [0,0,0,0,1,0,0,0,0,0]
            if y[i] == 5:
                one_hot_y[i] = [0,0,0,0,0,1,0,0,0,0]
            if y[i] == 6:
                one_hot_y[i] = [0,0,0,0,0,0,1,0,0,0]
            if y[i] == 7:
                one_hot_y[i] = [0,0,0,0,0,0,0,1,0,0]
            if y[i] == 8:
                one_hot_y[i] = [0,0,0,0,0,0,0,0,1,0]
            if y[i] == 9:
                one_hot_y[i] = [0,0,0,0,0,0,0,0,0,1]

        cost_for_sample =  -(one_hot_y * np.log(softmax_scores))

        return cost_for_sample

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y,number_of_iterations = 50, learning_rate = 0.01):

        one_hot_y = [0]*len(y)
        for i in range(len(y)):
                if y[i] == 0:
                    one_hot_y[i] = [1,0,0,0,0,0,0,0,0,0]
                if y[i] == 1:
                    one_hot_y[i] = [0,1,0,0,0,0,0,0,0,0]
                if y[i] == 2:
                    one_hot_y[i] = [0,0,1,0,0,0,0,0,0,0]
                if y[i] == 3:
                    one_hot_y[i] = [0,0,0,1,0,0,0,0,0,0]
                if y[i] == 4:
                    one_hot_y[i] = [0,0,0,0,1,0,0,0,0,0]
                if y[i] == 5:
                    one_hot_y[i] = [0,0,0,0,0,1,0,0,0,0]
                if y[i] == 6:
                    one_hot_y[i] = [0,0,0,0,0,0,1,0,0,0]
                if y[i] == 7:
                    one_hot_y[i] = [0,0,0,0,0,0,0,1,0,0]
                if y[i] == 8:
                    one_hot_y[i] = [0,0,0,0,0,0,0,0,1,0]
                if y[i] == 9:
                    one_hot_y[i] = [0,0,0,0,0,0,0,0,0,1]

        for k in range(number_of_iterations):
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True) 
           
            d = np.dot(np.transpose(X),softmax_scores - one_hot_y)
            b = np.dot(([1]*len(X)),softmax_scores - one_hot_y)
            self.theta = self.theta - (learning_rate*d)
            self.bias = self.bias - (learning_rate*b)

        #TODO:
        return 0

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def plot_decision_boundary(model, X, y):

    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    return 0



################################################################################    

posnames = ['N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13','N14','N15','N16',
            'N17','N18','N19','N20','N21','N22','N23','N24','N25','N26','N27','N28','N29','N30','N31','N32',
            'N33','N34','N35','N36','N37','N38','N39','N40','N41','N42','N43','N44','N45','N46','N47','N48','N49',
            'N50','N51','N52','N53','N54','N55','N56','N57','N58','N59','N60','N61','N62','N63','N64']
X = pandas.read_csv("DATA/Digits/X_test.csv",names = posnames)
y = pandas.read_csv("DATA/Digits/y_test.csv",names = 'c')
a = LogisticRegression(64,10)
a.fit(X,y['c'],1000,0.001)
pred = a.predict(X)
confusion_matrix = np.array([[0]*10]*10)
k = 0
for i in range(len(y['c'])):
    confusion_matrix[int(y['c'][i])][pred[i]]+=1
                     
print("\n\nConfusion Matrix = ")
for i in range(10):
    for j in range(10):
        print(confusion_matrix[i][j]),
    print

accuracy = np.array([0.]*10)
print("\n\nnumber          accuracy          TP           FP          FN          TN")
for i in range(10):
    TP = float(confusion_matrix[i][i])
    FP = float(np.sum(confusion_matrix[i]) - TP)
    FN = float(np.sum(np.transpose(confusion_matrix[i])) - TP)
    TN = float(len(y)-TP-FP-FN)
    accuracy[i] = float((TP+TN)/(TP+TN+FP+FN))
    print i,"             ",round(accuracy[i]*100),"          ",TP,"        ",FP,"       ",FN,"      ",TN
print "average accuracy = ", (np.sum(accuracy)/10)*100
        