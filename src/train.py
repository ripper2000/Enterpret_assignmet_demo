
import dataset as ds
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.metrics import accuracy_score ,log_loss,balanced_accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np

def av_metric(y_true, y_pred):
  return (f1_score(y_true, y_pred,average='macro'))

class training:
    def __init__(self) -> None:
        pass
    
    def simpletransformer_train(df_train,df_test,folder_name):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        model_args = ClassificationArgs(num_train_epochs=3,overwrite_output_dir=True,train_batch_size=16,max_seq_length=64)

        # Create a ClassificationModel
        model = ClassificationModel(
            "bert","bert-base-cased",
            #"distilbert","distilbert-base-cased",
            #"roberta", "roberta-base",
            num_labels=3,
            args=model_args) 
        N_SPLITS = 5
        oofs = np.zeros((len(df_train)))
        preds = np.zeros((len(df_test),3))

        folds = StratifiedKFold(n_splits = N_SPLITS)
        feature_importances = pd.DataFrame()

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train, df_train['label'])):
            print(f'\n------------- Fold {fold_ + 1} -------------')
            ### Training Set
            train_df = df_train[['text_filt','aspect','label']].iloc[trn_idx]
            ### Validation Set
            eval_df = df_train[['text_filt','aspect','label']].iloc[val_idx]
            
            train_df.columns = ["text_a", "text_b", "labels"]
            eval_df.columns = ["text_a", "text_b", "labels"]
            

            model.train_model(train_df.reset_index())
            preds_val, raw_outputs = model.predict(eval_df[["text_a", "text_b"]].values.tolist())
            
            fold_score = av_metric(eval_df['labels'], preds_val)
            print(f'\nAV metric score for validation set is {fold_score}')
            oofs[val_idx] = preds_val
            
        oofs_score = av_metric(df_train['label'], oofs)
        print(f'\n\nAV metric for oofs is {oofs_score}')

    
