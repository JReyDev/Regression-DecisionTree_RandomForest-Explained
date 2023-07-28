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
