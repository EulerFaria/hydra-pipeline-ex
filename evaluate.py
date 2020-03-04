import pandas as pd
from sklearn.metrics import classification_report




def generate_report(log,train_target,train_pred,
                    test_target,test_pred):


    log.info("Generating Evaluation Report:")
    train = classification_report(train_target,train_pred,output_dict=True)
    train = pd.DataFrame(train)

    log.info("Train report:")
    log.info('\n \t'+ train.to_string().replace('\n', '\n\t'))

    test = classification_report(test_target,test_pred,output_dict=True)
    test = pd.DataFrame(test)

    log.info("Test report:")
    log.info('\n \t'+ test.to_string().replace('\n', '\n\t'))
    

    
 