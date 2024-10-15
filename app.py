import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()

model = pickle.load(open('model.pkl','wb'))
ms = pickle.load(open('minmaxscaler.pkl','wb'))

def main():
    # User input for features
    st.title("Crop Classification and Recommendation System")
    soil_type = st.selectbox("Soil Type", options=["Type 1", "Type 2", "Type 3"])
    temperature = st.number_input("Temperature (°C)")
    rainfall = st.number_input("Rainfall (mm)")
   
    if st.button("Classify"):
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        ph = request.form['Ph']
        rainfall = request.form['Rainfall']

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        return render_template('index.html',result = result)

    

main()