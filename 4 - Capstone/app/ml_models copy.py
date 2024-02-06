import pandas as pd
import datetime

class ml_models:
    def __init__(self, model):
        self.model = model

    def fit_model(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test, y_test):
        # Predict on the test set
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        self.score =  self.model.score(X_test, y_test)
        return self.score

    def return_model(self):
        return self.model

    def log_message(self, message: str):
        # Logging the results to a file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - Model: {self.model} | {message} |Performance: {self.score:} \n"

        # Write the log to a file (change 'logfile.txt' to your desired log file name)
        with open('logfile.txt', 'a') as file:
            file.write(log_message)



   