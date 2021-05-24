from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def MLPclassification(data, labels):
    model = MLPClassifier()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30)
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    report = classification_report(y_train, predicted)
    print("For MLP classifier:\n", report)

def MLPclassification(data, labels):
    model = GradientBoostingClassifier()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    report = classification_report(y_train, predicted)
    print("For Gradient Boosting classifier:\n", report)

def RandomForestClassification(data, labels):
    model = RandomForestClassifier()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    report = classification_report(y_train, predicted)
    print("For Random Forest classifier:\n", report)

def allClassifiers(data, labels):
    MLPclassification(data, labels)
    MLPclassification(data, labels)
    RandomForestClassification(data, labels)