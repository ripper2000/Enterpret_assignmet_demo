import os
import re
from nltk.corpus import stopwords


class dataloader:
    def __init__(self) -> None:
        pass

    def pyabsa_dataloader(self,df,df_type,folder_name):
        file_path = os.path.join(folder_name ,f"apc {df_type}.txt")
        file_ = open(file_path,"w")
        inference_path = os.path.join(folder_name ,f"apc {df_type}.inference")
        fout = open(inference_path, 'w', encoding='utf-8', newline='\n', errors='ignore')
        len_df = len(df)
        for i,txt in df.iterrows():
            if(i==len_df):
                end=""
            # print(end)
            else:
                end ="\n"
                txt= df.at[i, 'text_filt'].replace(df.at[i, 'aspect'] , "$T$")
            
            file_.write(txt.replace("\n","")+end)
            file_.write(df.at[i, 'aspect']+end)
            
            if(df.at[i,'label'] == 0):
                file_.write("Negative"+end)
                lab ="Negative"
            if(df.at[i,'label'] ==1):
                file_.write("Neutral"+end)
                lab ="Neutral"
            if(df.at[i,'label'] == 2):
                file_.write("Positive"+end)
                lab ="Positive"
                        
            sample = txt.strip().replace('$T$', '[ASP]{}[ASP]'.format(df.at[i, 'aspect']))
            fout.write(sample + ' !sent! ' + lab + '\n')

        fout.close()
        file_.close()
        return folder_name
        