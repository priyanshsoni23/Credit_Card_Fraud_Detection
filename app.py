from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define a global variable to hold the model
model = None

# Define the routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    global model  # Access the global model variable

    if request.method == 'POST':
        # Extract input values from the form
        type = int(request.form.get('type'))       
        amount = float(request.form.get('amount'))
        oldbalanceOrg = float(request.form.get('oldbalanceOrg'))
        newbalanceOrig = float(request.form.get('newbalanceOrig'))

        # Check if the model is not yet trained
        if model is None:
            df=pd.read_csv('D:/python/content/fraud_detection.csv')
            df=df.dropna()
            df.replace(to_replace=['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'],value=[2,4,1,5,3],inplace=True)
            X_train = []
            y_train = []
            # Iterate over rows of the DataFrame
            for index, row in df.iterrows():
                # Create a list to store values for the current row
                row_data = []
                for column in ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']:
                    # Append the value of the current column to the row data
                    row_data.append(row[column])
                # Append the row data list to the list of data rows
                X_train.append(row_data)
                y_train.append(row['isFraud'])
            # Convert the list of data rows to a NumPy array
            X_train = np.array(X_train)

            # Initialize and train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        # Make a prediction using the trained model
        prediction = model.predict([[type, amount, oldbalanceOrg, newbalanceOrig]])

        # Determine the fraud status based on prediction
        if prediction[0] == 1:
            is_fraud = "fraud" 
        else:
            is_fraud = "not fraud"

        return render_template('result.html', is_fraud=is_fraud)

if __name__ == "__main__":
    app.run(debug=True)
