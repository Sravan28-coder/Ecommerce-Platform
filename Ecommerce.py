import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data from CSV
df = pd.read_csv('ecommerce.csv')

# TF-IDF Vectorization of product descriptions
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['product_description'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Combine TF-IDF features with user behavior features
features = pd.concat([df[['user_id', 'rating']], tfidf_df], axis=1)
target = df['recommended']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict recommendation
def predict_recommendation(user_id, product_description, user_rating=5):
    tfidf_desc = tfidf_vectorizer.transform([product_description])

    # Ensure the feature DataFrame matches the training data
    tfidf_array = tfidf_desc.toarray()
    features_array = [[user_id, user_rating] + tfidf_array[0].tolist()]
    features_df = pd.DataFrame(features_array, columns=features.columns)

    # Predict
    return model.predict(features_df)[0]

# Example usage with multiple samples
examples = [
    {"user_id": 1, "description": "Great value for money"},
    {"user_id": 2, "description": "Not worth the price"},
    {"user_id": 3, "description": "Fantastic!"},
    {"user_id": 4, "description": "Very poor experience"},
    {"user_id": 5, "description": "Excellent quality"},
    {"user_id": 1, "description": "Could be better"},
    {"user_id": 2, "description": "Love it, highly recommend"},
    {"user_id": 3, "description": "Good product"},
    {"user_id": 4, "description": "Average quality"},
    {"user_id": 5, "description": "Not as expected"}
]

# Print predictions for each example
for example in examples:
    user_id = example["user_id"]
    description = example["description"]
    prediction = predict_recommendation(user_id, description)
    print(f"Recommendation for user {user_id} with description '{description}': {'Recommended' if prediction == 1 else 'Not Recommended'}")
