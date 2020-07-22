"""
Methods for performing a search over specified parameter values for an estimator, with the goal of increasing model performance.
"""

from sklearn.model_selection._search import BaseSearchCV
from collections.abc import Iterable
from bayes_opt import BayesianOptimization
import colorama
import numpy as np

class BayesianOptCV(BaseSearchCV):    
    """
    Bayesian search over specified parameter values for an estimator.        
    
    Parameters
    ----------
    estimator: estimator object.
        This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.

    param_bounds: dict 
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values. The min and max values will be used as boundaries to the search space

    int_params: list
        List of parameters names that must be cast as integers.

    n_iter: int 
        How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.

    init_points: int
        How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        
    pre_dispatch : int, or str, default=None
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
              

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        
    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.
        .. versionchanged:: 0.20
            Support for callable added.
            
    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState instance, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
        .. versionadded:: 0.19
        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }
        NOTE
        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.
        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.
        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    """

    def __init__(self, estimator, param_bounds, int_params=[], n_iter=25, init_points=5, 
                 scoring=None, n_jobs=None, iid='deprecated', refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score=np.nan,
                 return_train_score=False):
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.init_points = init_points
        self.random_state = random_state
        self.int_params = int_params
        # Ensuring that the print statements are going to be properly formatted
        colorama.init()
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

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

    def _run_search(self, evaluate_candidates):        
        """Search n_iter candidates from param_bounds using the Bayesian Optimization method"""
        def _cv_wrapper(**kwargs):
            params = self._get_corrected_params(**kwargs)
            results =  evaluate_candidates([params])

            if self.multimetric_:
                if self.refit is not False and (
                        not isinstance(self.refit, str) or
                        # This will work for both dict / list (tuple)
                        self.refit not in scorers) and not callable(self.refit):
                    raise ValueError("For multi-metric scoring, the parameter "
                                    "refit must be set to a scorer key or a "
                                    "callable to refit an estimator with the "
                                    "best parameter setting on the whole "
                                    "data and make the best_* attributes "
                                    "available for that metric. If this is "
                                    "not needed, refit should be set to "
                                    "False explicitly. %r was passed."
                                    % self.refit)
                else:
                    refit_metric = self.refit
            else:
                refit_metric = 'score'

            return results["mean_test_%s" % refit_metric][-1]
        
        pbounds = dict()
        for param in self.param_bounds:
            if not isinstance(self.param_bounds[param], Iterable):
                self.param_bounds[param] = [self.param_bounds[param]]
            pbounds[param] = (min(self.param_bounds[param]), max(self.param_bounds[param]))

        optimizer = BayesianOptimization(
            f= _cv_wrapper,
            pbounds= pbounds,
            random_state=self.random_state,
            verbose=2
        )

        with np.errstate(invalid='ignore'):  # Ignore zero division warning caused by parameters with constant value
            optimizer.maximize(n_iter=self.n_iter, init_points=self.init_points)
