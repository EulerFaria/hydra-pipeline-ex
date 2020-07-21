"""
Methods for performing a search over specified parameter values for an estimator, with the goal of increasing model performance.
"""

from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from collections.abc import Iterable
from bayes_opt import BayesianOptimization
import numpy as np

class BayesianOptCV(BaseEstimator, TransformerMixin):    
    """
    Bayesian search over specified parameter values for an estimator.        
    
    Parameters
    ----------
    estimator: estimator object.
        This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.

    param_grid: dict 
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values. The min and max values will be used as boundaries to the search space

    int_params: list
        List of parameters names that must be cast as integers.

    n_iter: int 
        How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.

    init_points: int
        How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    scoring: str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
        If None, the estimator’s score method is used.

    cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

    """
    def __init__(self, estimator, param_grid, int_params, n_iter=10, init_points=2, scoring=None, cv=None, random_state=None, **kwargs):
        self.best_estimator_=estimator
        self.best_params_ = None
        self.best_score_ = None
        self.param_grid = param_grid
        self.scoring = scoring
        self.int_params = int_params
        self.cv = cv
        self.n_iter=n_iter
        self.init_points=init_points
        self.cv_results_ = {'std':[], 'mean':[]}
        self.random_state=random_state

    def _get_corrected_params(self, **kwargs):
        """
        Receives a dictionary and cast to integers all parameters specified on ``int_params``.
        """
        params = dict()
        for parameter in kwargs:
            if parameter in self.int_params:
                params[parameter] = int(kwargs[parameter])
            else:
                params[parameter] = kwargs[parameter]
        return params

    def _cv_score(self, X, y=None, **kwargs):
        """
        Returns a cross-validation score for a set of parameters.
        """
        params = self._get_corrected_params(**kwargs)
        self.best_estimator_.set_params(**params)
        cross_val = cross_val_score(self.best_estimator_, X, y, scoring=self.scoring, cv=self.cv)
        self.cv_results_['std'].append(cross_val.std())
        self.cv_results_['mean'].append(cross_val.mean())
        return cross_val.mean()

    def fit(self, X, y=None):
        """
        Run fit with bayesian optimization of the parameters

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)    
            Training vector, where n_samples is the number of samples and n_features is the number of features.
       
        y: array-like of shape (n_samples, n_output) or (n_samples,), default=None
            Target relative to X for classification or regression; None for unsupervised learning.

        """
        def _cv_wrapper(**kwargs):
            return self._cv_score(X, y, **kwargs)

        pbounds = dict()
        for param in self.param_grid:
            if not isinstance(self.param_grid[param], Iterable):
                self.param_grid[param] = [self.param_grid[param]]
            pbounds[param] = (min(self.param_grid[param]), max(self.param_grid[param]))

        optimizer = BayesianOptimization(
            f= _cv_wrapper,
            pbounds= pbounds,
            random_state=self.random_state,
            verbose=2
        )
        with np.errstate(invalid='ignore'):  # Ignore zero division warning caused by parameters with constant value
            optimizer.maximize(n_iter=self.n_iter, init_points=self.init_points)

        best_params = self._get_corrected_params(**optimizer.max['params'])
        self.best_params_ = best_params
        self.best_estimator_.set_params(**best_params)
        self.best_estimator_.fit(X, y)

        best_iter = np.argmin(self.cv_results_['mean'])
        self.best_score_ = (self.cv_results_['mean'][best_iter], self.cv_results_['std'][best_iter])

        return self

    def predict(self, X, y=None):
        """
        Generates prediction using optimized parameters

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)    
            Training vector, where n_samples is the number of samples and n_features is the number of features.
       
        y: array-like of shape (n_samples, n_output) or (n_samples,), default=None
            Target relative to X for classification or regression; None for unsupervised learning.

        """
        return self.best_estimator_.predict(X)