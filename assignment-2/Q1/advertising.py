# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


class Advertising:

    def __init__(self):
        self.input_file_path = './../../resource/advertising.csv'

    def execute(self):
        # Read data
        missing_value_formats = ['n.a', ',', '?', 'NA', 'n/a', 'N.A', '.', 'nan', 'NAN', '--', '0', '-', '_']
        df = pd.read_csv(self.input_file_path, na_values=missing_value_formats)
        df.head()

        # Check for null values
        df.isnull().sum()

        # Drop the columns that do not help in the model building df.drop(["Timestamp","City","Country","Ad Topic Line"],axis=1,inplace=True)
        # Split the features and the labels
        df.drop(["Ad Topic Line", "City", "Timestamp", "Country"],axis=1, inplace= True)
        x= df.drop(["Clicked on Ad"], axis=1)
        x.head()
        y = df["Clicked on Ad"]
        y.head()
        # Splitting the train and test data
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

        # Implementing the classifier
        model = Sequential()
        # Input layer
        model.add(Dense(128, activation='relu', input_shape=(5,)))
        # Hidden layer
        model.add(Dense(64, activation='relu'))
        # Hidden layer
        model.add(Dense(32, activation='relu'))
        # Hidden layer
        model.add(Dense(32, activation='relu'))
        # Output layer
        model.add(Dense(1, activation='sigmoid'))

        # Compiling the ANN
        model.compile(loss='binary_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        # Fitting the ANN
        model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)
        # Predict the values
        y_pred = model.predict(x_test)
        # Accuracy
        score = model.evaluate(x_test, y_test, verbose=0)
        print(score[1])


if __name__ == "__main__":
    ob = Advertising()
    ob.execute()