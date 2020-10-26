# import ntk libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant modules from sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """
    load the data we have prepared in ETL pipeline
    
    INPUT : database location 
    OUTPTU : 
    Output
        X : message column
        Y : dataframe with all categories
        category_names : List of categories name
    
    """
    
    # loading data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('vk_disasterResponse',engine)
    
    # As per our analysis in notebook, we know child_alone doesn't have any information, lets drop it
    df=df.drop(['child_alone'],axis=1)
            
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    # This will be used in classification_report
    category_names = Y.columns 
    return X, Y, category_names
    

def tokenize(text):
    """
    tokenize the text function
    
    input : message text
    output : returns the clean tokenize words form clean text
    """
    
    # replacing the urls text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # find all URLs from given message
    detected_urls = re.findall(url_regex, text)
    
    # replacing the URLs with strings in URL place holder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Extract the word tokens from the message
    tokens = nltk.word_tokenize(text)
    
    #lemmmatizing the tokenize words
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

# building a verb extractor function

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    This function will rxtract the part of verb and will create new feature for ML classifier
    """

    def starting_verb(self, text):
        """
        tokenize by sentences. It will tokenize each sentence into words and tag part of speech 
        and return true if the first word is an appropriate verb or RT for retweet
        INPUT : self and message
        OUTPUT : true and false based on approprite verb 
        
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
        
    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        """
        returns self instance  which is needed for piprline
        """
        return self

    def transform(self, X):
        """
        applying starting_verb function to all values in X
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    This will build the model pipeline
    
    Returns:
    ML pipeline which will process the input message and apply a classifier
    
    
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters_grid = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [10, 20, 40]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters_grid, scoring='f1_micro', n_jobs= 1)
    
    return cv
        


def evaluate_model(model, X_test, Y_test, category_names):
    
    ''' 
    
    Evaluate the model performance by f1 score, precision and recall
    Parameters:
    model: a ML model
    X_test: message from test set
    Y_test: category value from test set
    category_names: the names of the categories
    Return:
    None
    '''

    y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=y_pred, 
                          index=Y_test.index, 
                          columns=category_names)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    
    '''
    
    This will save the model on a given location
    
    Parameters:
    model: ML model 
    model_filepath: location where model will be saved
    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()