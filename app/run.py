import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

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


def tokenize(text):
    """
    tokenize the text function
    
    input : message text
    output : returns the clean tokenize words form clean text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/disasterresponse.db')
df = pd.read_sql_table('vk_disasterResponse', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    ### data for visualizing category counts.
    x_sums = df.iloc[:, 4:].sum()
    y_names = list(x_sums.index)
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # graph 2
        {
            'data': [
                Bar(
                    x=y_names,
                    y=x_sums,
                )
            ],

            'layout': {
                'title': 'Distribution of labels/categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {

                }
            }
        }
                
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=8000, debug=True)


if __name__ == '__main__':
    main()