from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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
        # print(request.form)
        # Extract input values from the form
        type = int(request.form.get('type'))
        amount = float(request.form.get('amount'))
        oldbalanceOrg = float(request.form.get('oldbalanceOrg'))
        newbalanceOrig = float(request.form.get('newbalanceOrig'))

        # Check if the model is not yet trained
        if model is None:
            # Generate example data for demonstration (replace with your dataset)
            X_train = np.array([
               [1, 100.0, 50.0, 50.0],
                [0, 200.0, 150.0, 50.0],
                [1, 300.0, 200.0, 100.0]
            ])
            y_train = np.array([0, 1, 0])  # Example target labels (0: Not Fraud, 1: Fraud)

            # Initialize and train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        # Make a prediction using the trained model
        prediction = model.predict([[type, amount, oldbalanceOrg, newbalanceOrig]])
        print(prediction)
        # Determine the fraud status based on prediction
        if prediction[0] == 1:
            is_fraud = "fraud"  # Use a consistent variable name
        else:
            is_fraud = "not fraud"

        return render_template('result.html', is_fraud=is_fraud)

if __name__ == "__main__":
    app.run(debug=True)
