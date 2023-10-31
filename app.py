

from __future__ import division, print_function
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import os

from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time
import numpy as np

from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SpatialDropout1D

app = Flask(__name__) 
Bootstrap(app)
	

@app.route('/', methods=['GET'])
def index():
    
    return render_template('predict1.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv('sqli.csv', encoding='utf-16')
    
    #checking if dataset is empty
    sns.heatmap(data.isnull(), cmap='viridis')
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(data.corr(), annot=True)
    
    plt.figure(figsize=(8, 6))
    ax=sns.countplot(data['Label'])
    ax.set_xticklabels(['SQL Injection Attack', 'Normal'])
    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
    posts = vectorizer.fit_transform(data['Sentence'].values.astype('U')).toarray()
    
    transformed_posts=pd.DataFrame(posts)
    
    data=pd.concat([data,transformed_posts],axis=1)
    
    X=data[data.columns[2:]]
    
    y = data['Label']
    
    X
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from keras.models import Sequential
    from keras import layers
    from keras.preprocessing.text import Tokenizer
    from keras.wrappers.scikit_learn import KerasClassifier
    
    
    
    input_dim = X_train.shape[1]  # Number of features
    
    model = Sequential()
    model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(10,  activation='tanh'))
    model.add(layers.Dense(1024, activation='relu'))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])


    if request.method == 'POST':
        comment = request.form['comments']
        if comment == '':
            empty = "Blank Page, please fill in a message"
            return render_template('predict.html', empty = empty)
        data =[comment]
        
        vec =  vectorizer.transform(data).toarray()
        from keras.models import load_model
        mymodel = load_model('dynamic.h5')
        pred = mymodel.predict(vec)
    return render_template('predict.html', pred = pred)
       
        
    
         
        


if __name__=='__main__':
	
	app.run(debug=True)