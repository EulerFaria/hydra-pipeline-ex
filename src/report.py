import os
import pandas as pd 
import numpy as np
import logging
from sklearn.metrics import classification_report,f1_score

logger = logging.getLogger(__name__)

def generate_report(opt,X_test,y_test):
    """Create a performance report for the current experiment and 
    consolidate the information to a general report of all runs 

    Parameters
    ----------
    opt : sklearn.model_selection.Object
        A hyperparameter 
    X_test: numpy array or pandas Dataframe 
        Input test data
    y_test: numpy array or pandas Dataframe 
        Target test data
    """

    path = os.path.normpath(os.getcwd() + os.sep + os.pardir) 

    logger.info("Generating Evaluation Report:")
    
    pipeline = opt.best_estimator_

    res = classification_report(y_test,pipeline.predict(X_test),output_dict=True)
    res = pd.DataFrame(res)

    logger.info("Test report:")
    logger.info('\n \t'+ res.to_string().replace('\n', '\n\t'))
    
    f1 = f1_score(y_test,pipeline.predict(X_test),average='macro')
    
    steps= [*pipeline.named_steps]

    cv_mean ,cv_std = opt.best_score_,opt.cv_results_['std_test_score'][opt.best_index_]

    tmp= pd.DataFrame({"Scaling":[steps[0]],
                        "Model":[steps[1]],
                        "params":[opt.best_params_],
                        'CV Mean':[cv_mean],
                        'CV Std':[cv_std],
                        'Test dataset':f1,
                        })

    if os.path.exists(path +"/results.csv"):
        current_csv =pd.read_csv(path +"/results.csv")
        _= pd.concat([current_csv,tmp],ignore_index=True).to_csv(path +"/results.csv",index=False)    
    else:
        tmp.to_csv(path +"/results.csv",index=False)
