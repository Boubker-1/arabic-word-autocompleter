# Arabic word autocompleter
A word predictor based on Arabic corpus made using Deep Learning models

LSTM (Long Short Term Memory) architecture was used for this project as it helps remembering the necessary information for an accurate prediction.

The idea for this model is instead of directly predicting the completion of a word, it actually predicts the last word that can contain the incomplete string that we want to predict its completion.

This is a classification problem as we treat the word we predict as a class, and every sequence in the corpus lines the model trained on has a word that can be treated as a lable. We will a number of classes equal to the number of unique words in the corpus lines

The corpus used for training this model was made by Mourad Abbas (Khaleej-2004) and it has 4 categories: Economy, International News, Local News and Sports

Due to memory limits, this model was only trained on 50 documents from Economy category.

The dataset can be found here: https://sites.google.com/site/mouradabbas9/arabic-corpora/text-corpora

Streamlit was used for visualization. Use "pip install streamlit" to install it and "streamlit run streamlit_visualizer.py" to run the visualizer and start predicting!

Try to keep words exclusive to economy field for convenience.
