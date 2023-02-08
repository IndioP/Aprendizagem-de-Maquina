import random
import numpy as np
from collections import Counter

def euclidean(point1,point2)->float:
    
    return np.linalg.norm(point1 - point2)

def manhattan(point1,point2)->float:
    
    return np.linalg.norm(point1-point2)

def hamming(point1,point2)->float:

    return np.sum(np.not_equal(point1,point2))

def majority_winner(rank)->float:
    votes = Counter(rank)
    winner,_ = votes.most_common(1)[0]
    return winner

def random_winner(rank)->float:
    return random.choice(rank)

def unique_winner(rank)->float:
    votes = Counter(rank)
    if len(votes.values()) == 1:
        winner,_ = votes.most_common(1)[0]
        return winner
    else:
        return unique_winner(rank[:-1])