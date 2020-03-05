import pandas as pd
import logging
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

def generate_report(train_target,train_pred,
                    test_target,test_pred):

    logger.info("Generating Evaluation Report:")
    train = classification_report(train_target,train_pred,output_dict=True)
    train = pd.DataFrame(train)

    logger.info("Train report:")
    logger.info('\n \t'+ train.to_string().replace('\n', '\n\t'))

    test = classification_report(test_target,test_pred,output_dict=True)
    test = pd.DataFrame(test)

    logger.info("Test report:")
    logger.info('\n \t'+ test.to_string().replace('\n', '\n\t'))
    

    
 