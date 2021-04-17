import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class Parkinsons:

    def __init__(self):
        self.input_file_path = './../../resource/parkinsons.csv'

    def execute(self):
        df = pd.read_csv(self.input_file_path)
        #df["name"] = [float(str(i).replace("_", "")) for i in df["name"]]
        df.head()

        X = df.drop("status", axis=1)
        Y = df["status"]

        # Splitting Data into 70% Training data and 30% Testing Data:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=42)
        print(len(X_train)), print(len(X_test))

        k_model = KNeighborsClassifier(n_neighbors=5)
        k_model.fit(X_train, y_train)
        k_model.score(X_test, y_test)

        y_pred = k_model.predict(X_test)

        count_misclassified = (y_test != y_pred).sum()
        print('Misclassified samples in KNN: {}'.format(count_misclassified))


if __name__ == "__main__":
    ob = Parkinsons()
    ob.execute()