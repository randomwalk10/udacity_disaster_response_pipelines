import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ this function load data of messages and categories
    messages_filepath: file path to message data
    categories_filepath: file path to categories data
    return:
        df: data frame merged from messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """ this funcition perform data cleaning: 1) split
    `categories` into 36 columns with values converted into
    numbers; 2) remove row duplicates; 3) drop categories with single value.
    df: dataframe contains information about messages and categories
    return:
        df: a cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicate rows
    df = df.drop_duplicates()
    # drop categories with single value
    cat_to_drop = []
    for label in df.columns[4:]:
        if len(np.unique(df[label])) < 2:
            cat_to_drop.append(label)
    print("Categories with single class:", cat_to_drop)
    df.drop(cat_to_drop, axis=1, inplace=True)
    # return
    return df


def save_data(df, database_filename):
    """ this funciton save data into database(`sql`)
    df: data frame to be saved
    database_filename: database filepath to save to
    return: None
    """
    # create sql engine
    engine = create_engine('sqlite:///'+database_filename)
    # save to database
    df.to_sql('message_categories_database', engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
