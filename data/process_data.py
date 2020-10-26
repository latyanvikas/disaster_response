import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load the disaster message and categories data 
    
    INPUT 
    disaster_categories.csv
    disaster_message.csv
    
    output 
    dataframe - merged dataset from above two files
    
    """
    
    df_messages   = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = pd.merge(df_messages,df_categories,on='id')
    
    return df


def clean_data(df):
    
    """
    claen the categores dataset
    
    inout 
    categories data
    
    output 
    
    cleaned dataset
    """
    # create a dataframe for all the individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # getting the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # form the above list create the list of new columns
    # we will be using lambda function to apply the name for all the columns
    
    cat_colnames = row.apply(lambda x:x[:-2])
    
    # rename the columns of `categories`
    categories.columns = cat_colnames
    
    # set each value to be the last character of the string
    # convert column from string to numeric

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from the dataframe df
    df=df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe created above
    df = pd.concat([df,categories],axis=1)
    
    # related columns doesn't have valid response, lets replace 2 with 1 as majority of the rows has this value. It mightbe due to error
    df.loc[:,'related'] = df['related'].replace(2,1)
    
    # remove duplicates 
    df = df.drop_duplicates()
    
    return df

        
    

def save_data(df, database_filename):
    '''
    save the final dataframe to database
    
    arguments:
    
    df - input dataframe
    database_filename - database location
   
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('vk_disasterResponse', engine, index=False,if_exists='replace')  


def main():
    
    """
    Main function to perform the ETL taks
    1) it will will extract the data from csv files
    2) it will do the necessary transofrmation
    3) it will load the data into sqlite database
    
    this function will not return anyrhing
    
    """
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