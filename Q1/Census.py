import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Census:

    def __init__(self):
        self.input_file_path = '/Users/shraj/Documents/BITS/2nd sem/ml/adult.csv'
        self.df = None

    def readCSV(self):
        self.df = pd.read_csv(self.input_file_path)
        print(self.df.head())
        print(self.df.shape)

    def clean_data(self):
        assert pd.notnull(self.df).all().all()  # none of the rows have null value, except for the work class where it is ?
        self.df = self.df[self.df.workclass != '?'] # we are removing the rows having work class as ?
        print(self.df.head())

    def visualize_data_sets(self):
        # self.df.plot.scatter(x='age', y='workclass', title='Census Dataset')
        columns = self.df.columns.drop(['workclass'])
        # create x data
        x_data = range(0, self.df.shape[0])
        # create figure and axis
        fig, ax = plt.subplots()
        # plot each column
        for column in columns:
            ax.plot(x_data, self.df [column], label=column)
        # set title and legend
        ax.set_title('census Dataset')
        ax.legend()

    def extract_x_and_y(self):
        # x as all columns except income
        # y as income
        df_temp = self.df
        x_df = df_temp.drop(['income'], axis=1)
        y_df = df_temp[['income']]
        x_columns = x_df.columns
        y_columns = y_df. columns
        print(x_columns)
        print(y_columns)

    def split_data_into_training_and_testing_set(self):
        y = self.df.target
        X_train, X_test, y_train, y_test = train_test_split(self.df, y, test_size=0.2)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)


if __name__ == '__main__':
    ob = Census()
    ob.readCSV()
    ob.clean_data()
    ob.visualize_data_sets()
    ob.extract_x_and_y()
    ob.split_data_into_training_and_testing_set()

