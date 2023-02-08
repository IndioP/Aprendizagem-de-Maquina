import numpy as np
from utils import euclidean, manhattan, hamming, majority_winner, random_winner, unique_winner
        

class knn():
    def __init__(self,k:int,metric:float='euclidean',heuristic:float='majority'):
        """k is the number of neighbors; metric is the distance algorithm options {'euclidean', 'manhattan', 'hamming'}; heuristic {'majority', 'random', 'unique'}"""
        self.k = k
        self.set_chosen_metric(metric)
        self.set_chosen_heuristic(heuristic)
        
    
    def set_chosen_metric(self,metric)-> None:
        self.metric = metric
        if metric == "euclidean":
            self.alg = euclidean
        elif metric == 'manhattan':
            self.alg = manhattan
        elif metric == 'hamming':
            self.alg = hamming    

    def set_chosen_heuristic(self,heuristic)->None:
        self.heuristic = heuristic
        if heuristic == "majority":
            self.voting_scheme = majority_winner
        elif heuristic == "random":
            self.voting_scheme = random_winner
        elif heuristic == "unique":
            self.voting_scheme = unique_winner

    def fit(self,x:np.array,y:np.array)->None:
        "x is an array of features, y is the array of labels"
        self.x = x
        print(x.ndim)
        self.y = y

    def _classify(self,x)->float:
        rank = [y for _,y in sorted(zip(self.x,self.y),key=lambda item: self.alg(item[0],x))]
        return self.voting_scheme(rank[:self.k])
    
    def predict(self,new_x)->float:
        assert new_x.shape[-1] == self.x.shape[-1]
        pred = []
        if new_x.ndim > 1:
            for i in new_x:
                pred.append(self._classify(i))
        else:
            pred.append(self._classify(new_x))
        return pred


    def __str__(self):
        return f"KNN OBJECT WITH PARAMETERS -> {self.k = }, {self.metric = }, {self.heuristic = }".replace("self.","")


if __name__ =="__main__":
    obj = knn(7,heuristic="unique",metric='hamming')
    import pandas as pd
    
    df = pd.read_csv("pima-indians-diabetes.csv",names='1 2 3 4 5 6 7 8 label'.split())
    print(df.head())
    x = df.drop('label',axis=1)
    y = df['label']
    
    print(obj)
    obj.fit(x.to_numpy(),y.to_numpy())
    
    print("winner = ",obj.predict(x.to_numpy()))
    x = np.array([6,  148,  72, 35,    0,  33.6,  0.627,  50])
    print("winner = ",obj.predict(x))