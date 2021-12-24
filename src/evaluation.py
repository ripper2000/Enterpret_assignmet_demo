from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import pickle
from dataset import dataloader
import os
import re
from nltk.corpus import stopwords
import torch
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

    def eval_simple_transformer_text(self,text,aspect,model_path):
        try:
            model = ClassificationModel(
                'bert',model_path,
                use_cuda=torch.cuda.is_available(),
            )
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


if __name__ == "__main__":
    eval = eval()
    df = pd.read_csv("data/test.csv")
    model_path = "model/simple transformer/final"
    out_path = "data/result/test_preds.csv"
    # eval.eval_ST_dataframe(df,model_path,out_path)
    eval.eval_simple_transformer_text("improve your customer service and product availability","Customer service",model_path)