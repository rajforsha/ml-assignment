import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVR


class Diamonds:

    def __init__(self):
        self.input_file_path = './../resource/diamonds.csv'
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
        assert pd.notnull(self.df).all().all()
        print(self.df.columns[self.df.isnull().any()])

    def visualize_data_sets(self):
        # self.df.plot.scatter(x='age', y='workclass', title='Census Dataset')
        columns = self.df.columns.drop(['z'])
        # create x data
        x_data = range(0, self.df.shape[0])
        # create figure and axis
        fig, ax = plt.subplots()
        # plot each column
        for column in columns:
            ax.plot(x_data, self.df [column], label=column)
        # set title and legend
        ax.set_title('Diamonds Dataset')
        ax.legend()

    def extract_x_and_y(self):
        # x as all columns except z
        # y as z
        df_temp = self.df
        x_df = df_temp.drop(['z'], axis=1)
        y_df = df_temp[['z']]
        x_columns = x_df.columns
        y_columns = y_df. columns
        print(x_columns)
        print(y_columns)
        return x_columns, y_columns

    def label_encoding(self):
        label_encoder = preprocessing.LabelEncoder()
        for column in ['Unnamed: 0', 'carat', 'cut', 'color', 'clarity', 'depth', 'table','price', 'x', 'y']:
            self.df[column] = label_encoder.fit_transform(self.df[column])
        print(self.df.head())

    def split_data_into_training_and_testing_set(self):
        list_of_x_cols, y_col = self.extract_x_and_y()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[list_of_x_cols], self.df[y_col], test_size=0.2)
        # clf = svm.SVC(C=1).fit(self.X_train, self.y_train)
        # clf.score(self.X_test, self.y_test)

    def logistic_regression(self):
        model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
        model.fit(self.X_train, self.y_train.values.ravel())
        print(model.coef_)
        print(model.intercept_)
        model.predict_proba(self.X_test)
        model.score(self.X_train, self.y_train)
        confusion_matrix(self.y_train, model.predict(self.X_test))



if __name__ == '__main__':
    ob = Diamonds()

    # Import the csv data set
    ob.readCSV()

    # Identify the columns with missing values.
    # Fill the missing values with mean value for numerical attributes and mode value for categorical attributes.
    # "","carat","cut","color","clarity","depth","table","price","x","y","z"
    ob.clean_data()

    # Extract X as all columns except the last column and Y as last column.
    ob.extract_x_and_y()

    # Visualize the dataset.
    ob.visualize_data_sets()

    # Split the data into training set and testing set. Perform 10-fold cross validation.
    ob.label_encoding()
    ob.split_data_into_training_and_testing_set()

    # Train a Logistic regression model for the data set.
    ob.logistic_regression()

