# Regression-DecisionTree_RandomForest-Explained

#### This guide is only for purpose of understanding how a decision tree regressor works but does not go into detail on ensemble methods, bootstrapping, utility of models, etc.

#### We first start by creating the essential functions of the decision tree. The word “split” will be used in the context of nodes and what threshold is set to divide the data such as X > 1000.

```
def _best_split(self, X, y):
        best_feature_index = None
        best_threshold = None
        best_cost = np.inf
```

#### Our first function is called _best_split, it starts off by setting up some variables that we will use to decide our split.

#### best_feature_index is the columns index where our feature is in our data, by default it is set to None. This is to ensure that if a split is not possible then return None for the index.

#### best_threshold is our value that we will be using to make our split, i.e., X < 1000 then 100 is the threshold. By default it is set to None as no data has been passed.

#### best_cost is our variance/MSE of the data group to its average value when a split is made, WE ARE LOOKING FOR THE SPLIT WITH THE LOWEST COST. It is by default set to np.inf as any number is less than positive infinity, this sets the first split cost found to be the lowest until another lower cost is found.

```
for feature_index in range(X.shape[1]):
        possible_thresholds = np.unique(X[:, feature_index])
```
#### After setting the variables, we begin with a for loop that iterates through the number of columns in the training data by calling the X. Shape function and taking the second index returned [1] which represents columns of our data to create the number of loops the function should run.

#### We then set the variable possible_thresholds to the unique values of the selected feature_index (column), that we will use to try to find the split for the data.

```
for threshold in possible_thresholds:
        cost = self._split_cost(X, y, feature_index, threshold)
```
#### Now that we are starting at our first feature column and created the unique values of the column, we can now begin to find the best split for our data.

#### This inner loop’s purpose is to loop through all possible thresholds of our feature column using the threshold to split the data, and calculate the cost of our split with the _split_cost function to then compare to other splits. We are looking for the lowest cost split.

```
if cost < best_cost:
        best_cost = cost
        best_feature_index = feature_index
        best_threshold = threshold

        return best_feature_index, best_threshold

```
#### This is where we compare our calculated cost to other costs of other splits.

#### We start by checking if the cost calculated is less than our current best_cost, if the cost is lower, then set best_cost, best_feautre_index, and best_threshold, to our current cost, feature_index, threshold respectively. 

#### Then return the variables, best_feature_index and best_threshold, which will be used to set the parameters of our Nodes.


```
    def _split_cost(self, X, y, feature_index, threshold):

        left_indices = X[:, feature_index] < threshold

        right_indices = X[:, feature_index] >= threshold

        left_cost = self._cost(y[left_indices])

        right_cost = self._cost(y[right_indices])

        cost = left_cost + right_cost

        return cost

```

#### Next is our _split_cost function which takes our X,y, feature_index (column index), and threshold (value).

#### The purpose of this function is to divide the data by the feature column and threshold into left and right, calculate the cost of the left and right split groupings and add the costs together to find the total cost of the split and return it.


```
    def _cost(self, y):
        if len(y) == 0:
            return 0
        return np.sum((y - np.mean(y)) ** 2)
```


#### This is our cost function, and its purpose is to calculate the variance/MSE of the split.

#### It first starts by checking if the length is 0, no observed values exist, then the cost is 0, which means it leads to a leaf node.

#### If not, then take the sum of (all the values of y subtracted from its meaning) squared. 

#### We do not divide by the number of values like in a variance/MSE calculation as this does not change the result much and only adds unnecessary arithmetic.

### Tree Nodes
#### Now that we know our functions then we can start the class 

# Node class to represent nodes in the decision tree

```
class DecisionNode:
    def __init__(self, feature_index=0, threshold=0, value=None, left=None, right=None):
        self.feature_index = feature_index 
        self.threshold = threshold  
        self.value = value  
        self.left = left  
        self.right = right  
```
#### This class serves as our nodes of our Decision Tree, it will hold the variables:
#### feature_index serves as the “selector” of the column of our data.
#### threshold is our value at our node to create splits such as X < 1000, the threshold would be 1000.
#### Value serves as the prediction if the node becomes a leaf node (“the answer”) and can no longer split or other stopping conditions are met.


```
class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
```
#### Now we begin creating our DecisionTreeRegressor Class and initiliaze it with variables:

#### Root serves as the starting point of the tree, by default it is empty as no tree exists yet.
#### Min_samples_split is our minimum number of values left to make a split after splitting the group, default is 2.
#### Max_depth serves as the max “levels” or the number of splits, default is 2.

```
    def fit(self, X, y, node=None, depth=0):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if node is None:
            node = DecisionNode()
            self.root = node

        if (len(y) < self.min_samples_split) or (depth == self.max_depth):
            node.value = np.mean(y)
```

#### Now we define the fit method, which is how we train the model. It takes as input, X as feature training data and Y as the observed result.

#### The method first checks for data types and converts them accordingly to be used.

#### Variable node is set to None to always create a new tree every time fit is called with a starting depth of 0.

#### Class starts by checking if the node variable is None, which it is by default, to always start a new tree every time the method fit is called.

#### The statement then sets the variable, node, to an instance of the class DecisionNode() then sets self.root to node, this is our root node.

#### The method then checks if any of our stopping conditions have been met, if so then get the average of the remaining samples in Y and assign it to the nodes.value parameter.
