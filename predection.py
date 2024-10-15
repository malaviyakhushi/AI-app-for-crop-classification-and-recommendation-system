import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the Streamlit app
st.title("Crop Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the data
    st.subheader("Dataset")
    st.write(df)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Prepare data for training
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features
    y = df['label']  # Target (crop label)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader(f"Model Accuracy: {accuracy:.2f}")
    
    # Prediction section
    st.subheader("Make a Prediction")
    
    # Input fields for new data
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=500, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=500, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=500, value=50)
    temperature = st.number_input("Temperature", min_value=-10.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (in mm)", min_value=0.0, max_value=500.0, value=100.0)
    
    # Make prediction based on user input
    user_input = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(user_input)
    print("prediction -",prediction )
    # Display the prediction
    st.subheader(f"Predicted Crop: {prediction[0]}")

# Footer
st.write("Developed with Streamlit and Scikit-learn")
