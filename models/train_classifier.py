import sys

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib


def load_data(database_filepath):
    """ this funciton load database from the given filepath and
    parse it into X(features) and Y(classes) for later use
    database_filepath: path to database
    return:
        X: dataframe for features
        Y: dataframe for classes/categories
        category_names: a list of names for each category
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('message_categories_database', engine)
    # get X(features)
    X = df["message"].values
    # get Y(classes)
    # do some data cleaning to remove classes with single values
    feature_names = ["id", "message", "original", "genre"]
    df = df.drop(feature_names, axis=1)
    cls_to_drop = []
    for label in df.columns:
        if len(np.unique(df[label])) < 2:
            cls_to_drop.append(label)
    df = df.drop(cls_to_drop, axis=1)
    Y = df.values
    # get category_names
    category_names = df.columns.tolist()
    # return
    return X, Y, category_names


def tokenize(text):
    """ this function tokenize the input text
    text: input raw text
    return:
        clean_tokens: a list of cleaned words
    """
    # initialize word tokenizer and lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    # return
    return clean_tokens


def build_model():
    """ this function build a pipelined model for disaster
    response.
    refer to ../jupyter_notebook_workspace/ML Pipeline Preparation.ipynb
    for more details about how the model parameters are obtained
    return:
        model: a pipelined model
    """
    # build pipeline
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
            ('tfidf', TfidfTransformer(use_idf=False))
        ])),

        ('clf', MultiOutputClassifier(SGDClassifier(
            loss='log', n_jobs=-1, random_state=42, alpha=0.0001, tol=0.001)))# use SVM
    ])
    # return
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ this funciton evaluate the input model, based on test data, by
    scores like accuracy, precision, recall, and f1
    model: a machine learning model
    X_test: test data of features
    Y_test: test data of classes
    category_names: a list names for classes
    return: None
    """
    # predict on the test data
    Y_pred = model.predict(X_test)
    # evaluate
    list_accuracy_scores = []
    for i, category in enumerate(category_names):
        target_names = [category+"_"+str(val) for val in np.unique(Y_test[:, i])]
        list_accuracy_scores.append(
            accuracy_score(Y_test[:, i], Y_pred[:, i]))
        print(classification_report(Y_test[:, i], Y_pred[:, i], target_names=target_names))
    print("average accuracy score is", sum(list_accuracy_scores)/len(list_accuracy_scores))


def save_model(model, model_filepath):
    """ this function save the input model to a give path
    model: model to be saved
    model_filepath: path to save the model into
    return: None
    """
    joblib.dump(model, model_filepath, compress=1)


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
