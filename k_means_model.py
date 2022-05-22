import numpy as np
from tqdm import tqdm

class Model(object):
    """
    Abstract model object. 
    """
    
    def fit(self, X, Y, iterations):
        """ 
        Fit the model
    
        Args:
            X: TODO: detemrine X type.
            Y: TODO: determine Y type.
            iterations: Int. Sets number of training iterations.
        """
        raise NotImplementedError()

    def predict(self, X, Y):
        """ 
        Predict network traffic type 
        
        Args: 
            X: TODO: detemrine X type.
            Y: TODO: determine Y type. 

        Returns:
            A dense array of ints with shape [num_examples]    
        """
        raise NotImplementedError()
     
class K_Mediods(object):

    def fit(self, X, Y, k, iterations):
        """ 
        Fit the K-Means model. And classify each cluster based on the labels. 
        
        Args:
            X: TODO: determine X type
            Y: TODO: determine Y
            k: Int. Number of clusters.
            iterations: Int. Number of iterations
        """

        for i in range(iterations):
            # E: Assign each example to the closest cluster
            x = 1
            # M: Compute mediod for each cluster.   

        # Classify each cluster based on the most common label.
        pass

    def predict(self, X):
        """ 
        Predict network traffic type
        
        Args: 
            X: TODO: determine X type
            Y: TODO: determine Y type
        Returns:
            A dense array of ints with shape [num_examples]
        """
        raise NotImplementedError()
    

