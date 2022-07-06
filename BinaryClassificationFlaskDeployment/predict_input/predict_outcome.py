from sklearn.ensemble import RandomForestClassifier
from model import X_train, y_train

class PredictOutcome:
    class_name = {0: "No, Your credit card is not at Risk", 1: "Yes, Your credit card is at Risk"}
    def __init__(self, input_value):
        self.input_value = input_value

    def predict(self):
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        result = rfc.predict(X= self.input_value) # o/p -> [0] or [1]
        # result = ''.join(result) join does not connect int list
        return self.class_name[result[0]]