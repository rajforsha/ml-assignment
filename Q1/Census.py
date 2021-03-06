import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


class Census:

    def __init__(self):
        self.input_file_path = './../resource/adult.csv'
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None

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
        return x_columns, y_columns

    def label_encoding(self):
        label_encoder = preprocessing.LabelEncoder()
        for column in ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']:
            self.df[column] = label_encoder.fit_transform(self.df[column])

        print(self.df.head())

    def split_data_into_training_and_testing_set(self):
        list_of_x_cols, y_col = self.extract_x_and_y()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[list_of_x_cols], self.df[y_col], test_size=0.2)

    def gaussianNB_model(self):
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train.values.ravel())
        y_pred = gnb.predict(self.X_test)
        self.print_accuracy_for_model_created("gaussianNB_model", metrics.accuracy_score(self.y_test, y_pred))
        self.print_confusion_matrix(confusion_matrix(self.y_test, y_pred))

    def multinomialNB_model(self):
        clf = MultinomialNB()
        clf.fit(self.X_train, self.y_train.values.ravel())
        y_pred = clf.predict(self.X_test)
        self.print_accuracy_for_model_created("multinomialNB_model", metrics.accuracy_score(self.y_test, y_pred))
        self.print_confusion_matrix(confusion_matrix(self.y_test, y_pred))

    def plot_decision_boundary(self):
        ""

    def print_accuracy_for_model_created(self, model_name, accuracy):
        print("Accuracy for the model:"+ model_name + "is", accuracy)

    def print_confusion_matrix(self, matrix):
        print(matrix)



if __name__ == '__main__':
    ob = Census()

    # Import the csv data set
    ob.readCSV()

    # Identify the presence of missing values, fill the missing values with mean for numerical attributes and mode value for categorical attributes.
    ob.clean_data()

    # Visualize the data set.
    ob.visualize_data_sets()

    # Extract X as all columns except the Income column and Y as Income column.
    ob.extract_x_and_y()

    # Model the classifier using GaussianNB and MultinomialNB
    ob.label_encoding()
    ob.split_data_into_training_and_testing_set()
    ob.gaussianNB_model()
    ob.multinomialNB_model()

    # Plot the decision boundary, visualize training and test results
    ob.plot_decision_boundary()
