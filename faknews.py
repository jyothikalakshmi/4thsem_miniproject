from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Load true news dataset and add a label column
true_df = pd.read_csv('True.csv')
true_df['label'] = 0

# Load fake news dataset and add a label column
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 1

# Combine the datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Train the Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input from the form
        user_input = request.form['news_article']
        print(f"User Input: {user_input}")
        
        # Load the trained model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Make prediction using the model
        prediction = model.predict([user_input])
        print(f"Prediction: {prediction}")
        
        # Determine the result based on prediction
        result = "The news is likely to be true." if prediction[0] == 0 else "The news is likely to be fake."

        # Return the result to the user
        return render_template('result.html', prediction=result)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please check the server logs.", 500

if __name__ == "__main__":
    app.run(debug=True)
