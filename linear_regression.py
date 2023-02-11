import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns

def draw_lm(x,y,y_pred):
    df = pd.DataFrame({"x":x,"y":y,"pred":y_pred})
    sns.lmplot(df,x='x',y='y')
    plt.plot(x,y_pred,color='green')

    

class linear_regression():
    def __init__(self,lr,debug=False):
        self.lr = lr
        #self.lr = 0.00000005
        self.debug = debug
        
    def fit(self,x:np.array,y:np.array):
        self.intercept = np.zeros(1)
        if x.ndim > 1:
            self.slope = np.zeros(x.shape[-1])
        else:
            self.slope = np.zeros(1)
        
        
        for i in range(5000):
            j_theta = np.zeros(self.slope.shape)
            for item_x, item_y in zip(x,y):
                
                
                y_hat = (self.slope * item_x)+self.intercept
                j_theta+=(y_hat-item_y)**2
            
            j_theta/=2*len(y)
            
            acum_intercept = np.zeros(1)
            acum_slope = np.zeros(self.slope.shape)
            errors = []
            for item_x, item_y in zip(x,y):
                
                y_hat = (self.slope * item_x)+self.intercept
                acum_intercept+=(y_hat-item_y)
                errors.append(y_hat-item_y)
                acum_slope+=(y_hat-item_y)*item_x
            #import pdb; pdb.set_trace()
            self.intercept = self.intercept - (acum_intercept*self.lr)*(1/len(y))
            self.slope = self.slope - ((self.lr)*(1/len(y)) * acum_slope)
            if self.debug:
                print("iteration=>",i,"theta0= ",self.intercept,"theta1=",self.slope)
                print("j_theta ===>",j_theta)
                print("errors =>",errors)
                y_pred = [(item_x*self.slope)+self.intercept for item_x in x]
                if i %1000 == 0:
                    draw_lm(x,y,y_pred)
        if self.debug:
            plt.show()
    
    def predict(self,x):
        if len(x)>1:
            y_hat = []
            for item_x in x:
                y_hat.append((item_x*self.slope)+self.intercept)
            return y_hat
        else:
            return [(x*self.slope)+self.intercept]

if __name__ == "__main__":
    #df = pd.DataFrame({"x":[2158.70,1708.30,2165.20,2053.50,1753.70],
    #                  'y':  [15.50,19.00,13.00,18.00,21.50]})
    #df = pd.DataFrame({"x":[58,62,60,64,67,70],
    #              'y':  [72,70,60,58,60,60]})#[60,60,58,60,70,72]})
    df = pd.DataFrame({"x":[58,62,60,64,67,70],
              'y':  [-60,-60,-58,-60,-70,-72]})
   
    model = linear_regression(0.000493,debug=True)
    model.fit(df.x.to_numpy(),df.y.to_numpy())
    print(model.predict(df.x.to_numpy()))
    print(model.predict(np.array([60])))