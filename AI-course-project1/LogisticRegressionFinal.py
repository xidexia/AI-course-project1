"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """

        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        """
        one_hot_y = [0]*len(y)
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        for i in range(len(y)):
            if y[i] == 1.00:
                one_hot_y[i] = [0,1]
            if y[i] == 0.00:
                one_hot_y[i] = [1,0]

        cost_for_sample =  -(one_hot_y * np.log(softmax_scores))
        

        """
        returns:
            cost: average cost per data sample
        """
        #TODO:
        return cost_for_sample

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y,number_of_iterations = 50, learning_rate = 0.01):
        """
        Learns model parameters to fit the data.
        """  
        for k in range(50):
            one_hot_y = [0]*len(y)
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True) 
            for i in range(len(y)):
                if y[i] == 1.00:
                    one_hot_y[i] = [0,1]
                if y[i] == 0.00:
                    one_hot_y[i] = [1,0]
            d = np.dot(np.transpose(X),softmax_scores - one_hot_y)
            b = np.dot(([1]*len(X)),softmax_scores - one_hot_y)
            self.theta = self.theta - (0.001*d)
            self.bias = self.bias - (0.001*b)

        #TODO:
        return 0

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################    
posnames = ['X','Y']
whichclass = ['whichclass']
data = pandas.read_csv("DATA/Linear/X.csv",names = posnames)
X_list = data.X.tolist()
Y_list = data.Y.tolist()
dataset = [data.X.tolist(),data.Y.tolist()]
X = np.transpose(dataset)
data = pandas.read_csv("DATA/Linear/y.csv",names = whichclass)
y = data.whichclass.tolist()
a = LogisticRegression(np.ndim(X),2)
a.fit(X,y,10000,0.001)
plot_decision_boundary(a,X,y)


