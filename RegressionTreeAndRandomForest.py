import numpy as np
import pandas as pd
#Reduce variance is the goal, reduce distance


# Node class to represent nodes in the decision tree
class DecisionNode:
    def __init__(self, feature_index=0, threshold=0, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature split
        self.value = value  # Value of the node if it is a leaf in the tree
        self.left = left  # 'Left' subtree for values < threshold
        self.right = right  # 'Right' subtree for values >= threshold

# Decision tree class
class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):
        # Initializing the tree as an empty node
        self.root = None

        # Stopping conditions for recursion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _best_split(self, X, y):
            best_feature_index = None
            best_threshold = None
            best_cost = np.inf

            # Iterate through all features
            for feature_index in range(X.shape[1]):
                # Check all possible thresholds
                possible_thresholds = np.unique(X[:, feature_index])
                ##print ('possible thresholds: ',possible_thresholds)
                
                # Evaluate all thresholds
                for threshold in possible_thresholds:
                    cost = self._split_cost(X, y, feature_index, threshold)
                    ##print ('current cost: ',cost)
                    ##print ('current threshold: ',threshold)

                    if cost < best_cost:
                        best_cost = cost
                        best_feature_index = feature_index
                        best_threshold = threshold
                        ##print('best cost: ', best_cost,'\n')

                ##print('best threshold: ',best_threshold)
                ##print('best feature index: ',best_feature_index)

            return best_feature_index, best_threshold
    
        # Method to calculate the cost of a split (total variance)
    def _split_cost(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] < threshold
        ##print('left indice: ',left_indices)
        right_indices = X[:, feature_index] >= threshold
        ##print('right indices', right_indices)
        
        #print ('left indices: ', left_indices)

        # Calculate the cost as the sum of the squared deviation from the mean
        left_cost = self._cost(y[left_indices])
        ##print('left cost: ',left_cost)
        right_cost = self._cost(y[right_indices])
        ##print('right cost: ',right_cost)

        # Total cost is the sum of left and right cost
        cost = left_cost + right_cost
        ##print('final cost: ',cost)
        return cost

    # Calculate the variance of y values, sum of squared differences from the mean, no dividing to save computation power
    def _cost(self, y):
        if len(y) == 0:
            return 0
        return np.sum((y - np.mean(y)) ** 2)

    # Method to train the decision tree
    def fit(self, X, y, node=None, depth=0):
        # Convert pandas DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if node is None:
            # Create a root node if it doesn't exist
            node = DecisionNode()
            self.root = node

        # If the number of samples is less than the minimum samples needed for a split
        # or if the max depth has been reached, make this node a leaf node and assign it the mean value of its group
        if (len(y) < self.min_samples_split) or (depth == self.max_depth):
            node.value = np.mean(y)
            return

        # Determine the best split
        feature_index, threshold = self._best_split(X, y)

        # If no valid split is possible, make this a leaf node as well
        if feature_index is None or threshold is None:
            node.value = np.mean(y)
            return

        # Assign the best feature index and threshold to this node
        node.feature_index = feature_index
        node.threshold = threshold

        # Create left and right child nodes
        node.left = DecisionNode()
        node.right = DecisionNode()

        # Create boolean mask for split
        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        # Perform recursion
        self.fit(X[left_indices], y[left_indices], node.left, depth + 1)
        self.fit(X[right_indices], y[right_indices], node.right, depth + 1)


    


    

    # Method to predict the target variable for a given feature vector
    def predict(self, X, node=None):
        if node is None:
            node = self.root

        # If we reach a leaf node, return the value of the node
        if node.value is not None:
            return node.value

        # Depending on the feature of X at the feature_index, decide whether to go left or right
        ##print ('is' ,X[node.feature_index] ,'less then: ',node.threshold)
        ##print ('predict feature index: ',node.feature_index, '\n')
        #print ('predict threshold: ', node.threshold)

        if X[node.feature_index] < node.threshold:
            return self.predict(X, node.left)
        else:
            return self.predict(X, node.right)


import numpy as np

# Trading volume (in '000s)
volume = np.array([1000, 1200, 1500, 2000, 1800, 2100, 1900, 1600])

# Opening price (in $)
open_price = np.array([50, 52, 51, 53, 54, 56, 55, 57])

# Closing price (in $)
close_price = np.array([51.49671415, 53.0617357 , 53.14768854, 56.52302986, 55.56584663, 57.86586304, 58.47921282, 59.36743473])

# Features
X = np.column_stack([volume, open_price])
# Target
y = close_price


# Initialize the DecisionTreeRegressor object
tree = DecisionTreeRegressor(min_samples_split=2, max_depth=5)


# Fit the model to the data
tree.fit(df, y)

# Predict the closing price for a new data point
new_data = np.array([3000, 70])  # New data point: volume = 3000 ('000s), opening price = $70
predicted_close_price = tree.predict(new_data)
print(predicted_close_price)


import numpy as np

class RandomForestRegressor:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        # Convert pandas DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


rf = RandomForestRegressor()
rf.fit(X,y)
rf.predict(new_data)