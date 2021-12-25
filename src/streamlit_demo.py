import streamlit as st
from io import StringIO 

from evaluation import eval

def run():
    
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
    if st.button("Predict"):
        eval_ = eval()

        output = eval_.eval_pyabsa_text(text =text.lower(),aspect =aspect.lower(),model_path='model/pyabsa/checkpoints')
        output = str(sent_dict[output]) # since its a list, get the 1st item
        st.success(f"The given phrase is {output} towards aspect {aspect} ")
        # st.balloons()
if __name__ == "__main__":
    run()
