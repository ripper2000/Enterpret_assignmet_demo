
import streamlit as st
from io import StringIO 

from evaluation import eval
# class FileReference:
#     def __init__(self, filename):
#         self.filename = filename

# @st.cache()
def load_model(pyabsa_path,st_path):
    
    eval_ = eval()
    pyabsa_model = eval_.load_models(pyabsa_path,st_path)
    return pyabsa_model

pyabsa_model ,st_model = load_model(pyabsa_path="model/pyabsa/checkpoints",st_path="model/simple transformer/final")

def run( ):
    
    st.title("Enterpret Internship Assignment 1.")
    st.header('Predicting sentiment for selected aspect in given phrase')
    

    text = st.text_area('Enter the phrase')#improve your customer service and product availability
    aspect = st.text_area('Aspect')#Customer service
    output = ""
    sent_dict = {
        0:"Negative",
        1:"Neutral",
        2:"Positive"
    }
    
    if st.button("Predict using pyabsa model"):
        eval_ = eval()

        output = eval_.eval_pyabsa_text(text =text.lower(),aspect =aspect.lower(),model=pyabsa_model)
        output = str(sent_dict[output]) # since its a list, get the 1st item
        st.success(f"The given phrase is {output} towards aspect {aspect} ")
        # st.balloons()
#temporarily disabled to reduce resource usage
    # if st.button("Predict using simple transformer model"):
    #     eval_ = eval()

    #     output = eval_.eval_ST_text(text =text.lower(),aspect =aspect.lower(),model=st_model)
    #     output = str(sent_dict[output[0]]) # since its a list, get the 1st item
    #     st.success(f"The given phrase is {output} towards aspect {aspect} ")
    #     # st.balloons()

if __name__ == "__main__":
    pyabsa_model ,st_model = load_model(pyabsa_path="model/pyabsa/checkpoints",st_path="model/simple transformer/final")
    run(pyabsa_model ,st_model)
