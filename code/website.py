import streamlit as st

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

news_df=pd.read_csv('train.csv')
news_df=news_df.fillna(' ')
news_df['content']=news_df['author']+" "+news_df['title']
#stemming
ps=PorterStemmer()
def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content) 
    return stemmed_content
news_df['content']= news_df['content'].apply(stemming)
X=news_df['content'].values
y=news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)
#model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

model=LogisticRegression()
model.fit(X_train,y_train)
train_y_pred=model.predict(X_train)
print("Train accuracy:",accuracy_score(train_y_pred,y_train))
test_y_pred=model.predict(X_test)
print("Train accuracy:",accuracy_score(test_y_pred,y_test))

#test
input_data=X_test[200]
prediction=model.predict(input_data)
if prediction[0]==1:
    print("Fake news")
else:
    print("Real news")
    
    
st.title('Fake News Detector')
input_text=st.text_input("Enter news Article")

def prediction(input_text):
    input_data=vector.transform([input_text])
    prediction=model.predict(input_data)
    return prediction[0]

if input_text:
    pred=prediction(input_text)
    if pred==1:
        st.write("The news is Fake!")
    else:
        st.write("The News is Real")