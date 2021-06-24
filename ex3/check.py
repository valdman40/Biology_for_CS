# TODO 1: implement WRF following the provided class API. You should support both, classification as
# well as regression (the type argument can be either "cat" or "reg"). You should use DecisionTreeClassifier
# and DecisionTreeRegressor as the underlying trees.

from collections import defaultdict


class WRF:
    def __init__(self, n_trees=100, max_depth=5, max_features=None, type="reg", weight_type="div", n_estimators=None):
        # def __init__(self, type,n_trees=100, max_depth=5, max_features=None, weight_type="div", n_estimators = None):
        # print('WRF init')
        '''
          init a WRF classifier with the following parameters:

          n_trees: the number of trees to use.
          max_depth: the depth of each tree (will be passed along to DecisionTreeClassifier/DecisionTreeRegressor).
          n_features: the number of features to use for every split. The number should be given to DecisionTreeClassifier/Regressor as max_features.
          type: "cat" for categorization and "reg" for regression.
          weight_type: the tree weighting technique. 'div' for 1/error and 'sub' for 1-error.
        '''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.type = type
        self.weight_type = weight_type
        self.n_estimators = n_estimators

    def fit(self, X, y):
        # print('WRF fit')
        '''
          fit the classifier for the data X with response y.
        '''
        # <Your Code if needed>
        self.trees = []
        self.weights = []
        for n in range(self.n_trees):
            tree = self.build_tree()
            self.trees.append(tree)
            X_tree, y_tree, X_oob, y_oob = self.bootstrap(X, y)
            tree.fit(X_tree, y_tree)
            weight = self.calculate_weight(tree, X_oob, y_oob)
            self.weights.append(weight)

        # Normalize the weights so they sum to 1
        # <Your code goes here>
        norm_weights = []
        weights_sum = sum(self.weights)
        for weight in self.weights:
            norm_cur_weight = weight / weights_sum
            norm_weights.append(norm_cur_weight)
        self.weights = norm_weights
        print('done fit')

    def build_tree(self):
        # print('WRF build_tree')
        tree = None
        if self.type == "cat":
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
        else:
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features)
        return tree

    def get_oob(self, X, y, X_tree, y_tree):
        # print('WRF get_oob')
        X_cols = []
        for col in X:
            X_cols.append(col)

        y_cols = []
        for col in y:
            y_cols.append(y_cols)

        # out of bag observations
        X_oob = pd.DataFrame(columns=X_cols)
        y_oob = pd.Series([], dtype=pd.StringDtype())
        count = 0
        for x, label in zip(X.values, y.values):
            is_x_found = False
            for idx in range(len(X_tree)):
                # CHECK IF TWO ARRAYS ARE EQUIVALENT
                cur = X_tree.values[idx]
                is_equal = np.array_equal(x, cur)
                if is_equal:
                    is_x_found = True
            if is_x_found is False:
                X_oob.loc[count] = x
                y_oob = y_oob.append(pd.Series([label]))
                count += 1

        return X_oob, y_oob

    def my_resample(self, X, y, n_samples):
        # print('WRF my_resample')
        X_cols = []
        for col in X:
            X_cols.append(col)

        y_cols = []
        for col in y:
            y_cols.append(y_cols)

        X_tree = pd.DataFrame(columns=X_cols)
        y_tree = pd.Series([], dtype=pd.StringDtype())
        for _ in range(n_samples):
            i = np.random.randint(n_samples)
            add_x = X.values[i]
            add_label = y.values[i]
            X_tree.loc[_] = add_x
            y_tree = y_tree.append(pd.Series([add_label]))

        return X_tree, y_tree

    def bootstrap(self, X, y):
        # print('WRF bootstrap')
        '''
          This method creates a bootstrap of the dataset (uniformly sample len(X) samples from X with repetitions).
          It returns X_tree, y_tree, X_oob, y_oob.
          X_tree, y_tree are the bootstrap collections for the given X and y.
          X_oob, y_oob are the out of bag remaining instances (the ones that were not sampled as part of the bootstrap)
        '''
        # <Your code goes here>
        # X_tree, y_tree = resample(X, y, replace=True, n_samples=len(X))  # prepare bootstrap sample
        X_tree, y_tree = self.my_resample(X, y, n_samples=len(X))  # prepare bootstrap sample
        X_oob, y_oob = self.get_oob(X, y, X_tree, y_tree)
        return X_tree, y_tree, X_oob, y_oob

    def calculate_weight(self, tree, X_oob, y_oob):
        # print('WRF calculate_weight')
        '''
          This method calculates a weight for the given tree, based on it's performance on
          the OOB instances. We support two different types:
          if self.weight_type == 'div', we should return 1/error and if it's 'sub' we should
          return 1-error. The error is the normalized error rate of the tree on OOB instances.
          For classification use 0/1 loss error (i.e., count 1 for every wrong OOB instance and divide by the numbner of OOB instances),
          and for regression use mean square error of the OOB instances.
        '''
        # < Your code goes here>

        sum_mis = 0
        oob_len = len(X_oob)
        # categorization
        if (self.type == "cat"):
            y_predictions = tree.predict(X_oob)
            for x, prediction in zip(X_oob, y_predictions):
                if x != prediction:
                    sum_mis += 1
            sum_mis = sum_mis / len(X_oob)

        # regression
        if (self.type == "reg"):
            pred = tree.predict(X_oob)
            pred = list(pred)
            sum_mis += mean_squared_error(pred, y_oob, squared=False)

        # check if 'div' or 'sub'
        if (self.weight_type == "div"):
            return 1 / sum_mis
        return 1 - sum_mis  # 'sub'

    def predict(self, X):
        '''
          Predict the label/value of the given instances in the X matrix.
          For classification you should implement weighted voting, and for regression implement weighted mean.
          Return a list of predictions for the given instances.
        '''
        # <Your code goes here>
        # regression
        if (self.type == "reg"):
            all_preds = np.zeros(len(X))
            for tree, cur_tree_weight in zip(self.trees, self.weights):
                pred = tree.predict(X) * cur_tree_weight
                all_preds = all_preds + pred
            prediction = all_preds / sum(self.weights)
            return prediction

        # categorization
        if (self.type == "cat"):
            return 5

        print('got to end without going into types')
        return 10

    def get_params(self, deep=True):
        # print('WRF get_params')
        out = dict()
        param_names = ['max_depth', 'n_estimators', 'max_features']
        for key in param_names:
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # print('WRF set_params')
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix

        for key, value in params.items():
            key, delim, sub_key = key.partition('__')

            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class WFRClassification(WFR):
    def __init__(self, n_trees=100, max_depth=5, max_features=None, weight_type="div", n_estimators=None):
        super().__init__(self, n_trees, max_depth, max_features, "cat", weight_type, n_estimators)


class WFRRegression(WFR):
    def __init__(self, n_trees=100, max_depth=5, max_features=None, weight_type="div", n_estimators=None):
        super().__init__(self, n_trees, max_depth, max_features, "reg", weight_type, n_estimators)
