import streamlit as st
import processing
st.title('Word autocompletion(arabic corpus)')

st.write('''This application is a demo result of our work in the project''')

text = st.text_input("Enter a sentence!") 

if st.button('Complete'):

    with st.spinner('Wait for prediction it take sometime, approximatly 5 seconds !'):
        if processing.predict(text):
            exit

    
    
    

    
