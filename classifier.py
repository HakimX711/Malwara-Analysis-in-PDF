# import necessary libraries and modules
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# import the feature_extraction function from the feature_extraction module
from feature_extraction import feature_extraction

# read the csv file and store the data in a Pandas DataFrame
df = read_csv('PDFMal.csv')

# assign the features (columns 0-20) to X and the target (column 21) to y
X = df.iloc[:, 0: 21]
y = df.iloc[:, 21]

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print message to indicate beginning of model evaluation
print("---Random Forest---")

# create a Random Forest classifier and fit it to the training data
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# use the trained model to make predictions on the test data
y_pred = clf.predict(X_test)

# evaluate the model's performance
acs = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# print the results of the evaluation
print("Accuracy:", acs)
print("\nConfusion Matrix:\n", cm)
print("Prediction",y_pred)
print(classification_report(y_test, y_pred))

