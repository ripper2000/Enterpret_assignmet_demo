from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import pickle
from dataset import dataloader
import os
import re
from nltk.corpus import stopwords
import torch
from pyabsa import APCCheckpointManager, ABSADatasetList, available_checkpoints
import spacy
from spacy.cli import download
print(download("en_core_web_sm"))


# device = torch.device('cpu')
class eval:
    def __init__(self) -> None:
        pass

    def simple_transformer_clean_text(self,text):
        stop_words = set(stopwords.words('english'))   
        text = str(text).lower()
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(pattern=r'https?://\S+|www\.\S+', repl=' ', string=text)
        text = re.sub(r'[^0-9A-Za-z]'," ",text)
        #text = [i for i in text.split(' ') if not i in stop_words]
        text = [w for w in text.split() if len(w) > 2]
        text = ' '.join(text)
        return text

    def simple_transformer_clean_aspect(self,text):
        stop_words = set(stopwords.words('english'))   
        #ps = PorterStemmer()
        text = str(text).lower()
        text = re.sub(pattern=r'https?://\S+|www\.\S+', repl=' ', string=text)
        text = re.sub(r'[^0-9A-Za-z]'," ",text)
        text = [i for i in text.split(' ') if not i in stop_words]
        #text = [ps.stem(w) for w in text if len(w) > 1]
        text = [w for w in text if len(w) > 1]
        text = ' '.join(text)
        return text

    def load_models(self ,pyabsa_path,st_path):
        pyabsa_model =APCCheckpointManager.get_sentiment_classifier(
                                                                checkpoint=pyabsa_path,
                                                                auto_device=True,  # Use CUDA if available
                                                                
                                                                )
        # st_model = ClassificationModel(
        #         'bert',st_path,
        #         use_cuda=torch.cuda.is_available(),
        #     )
        return pyabsa_model 
    def eval_ST_text(self,text,aspect,model):
        try:
            
            preds, raw_outputs = model.predict([text,aspect])
            return preds
        except Exception as e:
            print(e)
            return e, "Error"

    def eval_ST_dataframe(self,df,model_path,out_path):
        try:
            
            df["text"]  = df["text"].apply(lambda x: self.simple_transformer_clean_text(x))
            df["aspect"] = df["aspect"].apply(lambda x: self.simple_transformer_clean_aspect(x))
            model = ClassificationModel(
                'bert',model_path,
                use_cuda=torch.cuda.is_available(),
            )
            preds, raw_outputs = model.predict(df[["text","aspect"]].values.tolist())
            df["label"] = preds
            df.to_csv(out_path,index=False)
            return "Success"
        except Exception as e:
            print(e)
            return e
    def eval_pyabsa_text(self , text ,aspect ,model):
        try:
            text = text.lower().replace("\n","")
            # print(text)
            label_dict = {
                "Negative": 0,
                "Neutral": 1,
                "Positive":2
            }
            rev_dict = {value : key for (key, value) in label_dict.items()}
            # print(rev_dict)
            text_to_send = text.replace(aspect,"[ASP]"+aspect+"[ASP]")
            # print(text_to_send)
            
            result = model.infer(text_to_send,print_result=True)
            return label_dict[result[0]['sentiment'][0]]
        except Exception as e:
            print(e)
            return 

    def eval_pyabsa_dataframe(self,df,model_path,out_path):
        try:
            
            model = APCCheckpointManager.get_sentiment_classifier(
                                                                checkpoint=model_path,
                                                                auto_device=True,  # Use CUDA if available
                                                                
                                                                )
            df["label"] = df.apply(lambda row: self.eval_pyabsa_text(row.text, row.aspect,model),axis=1)
            df.to_csv(out_path,index=False)
            return "Success"
        except Exception as e:
            print(e)
            return e


if __name__ == "__main__":
    eval = eval()
    df = pd.read_csv("data/test.csv")
    model_path = "model/simple transformer/final"
    pyabsa_model_path ="model/pyabsa/checkpoints"
    out_path = "data/result/test_preds.csv"
    pyabsa_out_path = "data/result/test_pyabsa_preds.csv"
    eval.eval_pyabsa_dataframe(df,pyabsa_model_path,pyabsa_out_path)
    # eval.eval_simple_transformer_text("improve your customer service and product availability","Customer service",model_path)