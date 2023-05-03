from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load the saved model
filename = 'adaboost_model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Define the prediction function
def predict_diabetes(data):
    # Extract the features from the input data
    features = np.array(list(data.values())).reshape(1,-1)
    # Make predictions using the loaded model
    prediction = model.predict(features)
    return prediction

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)
    # Make predictions on the input data
    prediction = predict_diabetes(data)
    # Return the prediction as a JSON response
    response = {'prediction': int(prediction[0])}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
