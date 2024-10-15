import streamlit as st
import pandas as pd

# Title of the Streamlit app
st.title("Crop Recommendation Data Analysis")

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
    
    # Crop selection for filtering
    crop_list = df['label'].unique()
    selected_crop = st.selectbox("Select a crop to filter", crop_list)
    
    # Filter the dataframe based on selected crop
    filtered_df = df[df['label'] == selected_crop]
    
    # Display filtered data
    st.subheader(f"Data for {selected_crop}")
    st.write(filtered_df)
    
    # Visualization: Nutrient levels
    st.subheader("Nutrient Levels (N, P, K)")
    st.bar_chart(filtered_df[['N', 'P', 'K']])
    
    # Visualization: Environmental factors
    st.subheader("Environmental Conditions")
    st.line_chart(filtered_df[['temperature', 'humidity', 'ph', 'rainfall']])


