import streamlit as st
from joblib import load
import pandas as pd
import ast

# Load the RandomForest model and the scaler
rf_model = load('rf_model.joblib')
scaler = load('scaler.joblib')

rf_model_tod = load('tod_rf_model.joblib')
scaler_tod = load('tod_scaler.joblib')

# Function to load data with caching
@st.cache_data  # ensure the data is loaded only once
def load_data():
    return pd.read_csv('data2.csv')
def load_district():
    # Load the CSV data
    return pd.read_csv('districtneighborhoods.csv')
def load_tod():
    return pd.read_csv('tod.csv')
data = load_data()
district_data = load_district()
tod_data = load_tod()


# Calculate historical averages
@st.cache_data
def get_historical_averages(data,timeframe):
    return data.groupby([timeframe, 'cluster'])['incident_count'].mean().reset_index()

historical_averages = get_historical_averages(data,'month')
historical_averages_tod = get_historical_averages(tod_data,'time_of_day')
st.title("Incident Prediction")

# Input fields for month and cluster which could influence incident rates
month_value = st.slider("Select the month:", min_value=1, max_value=12, step=1)
#cluster_value = st.slider("Select the cluster (based on neighborhood characteristics):", min_value=0, max_value=4, step=1)
tod_value = st.slider("Select the hour of the day:", min_value=0, max_value=23, step=1)

districts = district_data['police_district'].unique()
district = st.selectbox("Select a district:", districts)

if district:
    neighborhoods = district_data[district_data['police_district'] == district]['analysis_neighborhood'].unique()
    neighborhood = st.selectbox("Select a neighborhood:", neighborhoods)



if st.button('Predict Incident Counts MONTH'):
    # Filter the historical averages based on the selected month and cluster
    if district and 'neighborhood' in locals():
        cluster_value = data[data['analysis_neighborhood'] == neighborhood]['cluster'].iloc[0]
        st.write(f"Selected Cluster: {cluster_value}")

    filtered_data = historical_averages[
        (historical_averages['month'] == f'2023-{month_value:02}') & 
        (historical_averages['cluster'] == cluster_value)
    ]
    

    if not filtered_data.empty:
        # Extract the first (and should be only) incident count value
        historical_incident = filtered_data['incident_count'].iloc[0]

        # Normalize the historical incident counts
        normalized_incidents = scaler.transform([[historical_incident]])[0][0]

        # Prepare the data for prediction
        input_data = pd.DataFrame({
            'month_encoded': [month_value],
            'cluster': [cluster_value],
            'normalized_incidents': [normalized_incidents]
        })

        # Predict using the RandomForest model
        prediction = rf_model.predict(input_data)[0]
        st.write(f"Predicted number of incidents: {prediction}")
    else:
        st.error("No historical data available for the selected month and cluster.")


if st.button('Predict Incident Counts TOD'):
    # Filter the historical averages based on the selected time of day and cluster
    if district and 'neighborhood' in locals():
        cluster_value = tod_data[tod_data['analysis_neighborhood'] == neighborhood]['cluster'].iloc[0]
        st.write(f"Selected Cluster: {cluster_value}")
    filtered_data_tod = historical_averages_tod[
        (historical_averages_tod['time_of_day'] == tod_value) & 
        (historical_averages_tod['cluster'] == cluster_value)
    ]

    if not filtered_data_tod.empty:
        # Extract the first (and should be only) incident count value
        historical_incident_tod = filtered_data_tod['incident_count'].iloc[0]

        # Normalize the historical incident counts
        normalized_incidents_tod = scaler_tod.transform([[historical_incident_tod]])[0][0]

        # Prepare the data for prediction
        input_data_tod = pd.DataFrame({
            'time_of_day': [tod_value],
            'cluster': [cluster_value],
            'normalized_incidents': [normalized_incidents_tod]
        })

        # Predict using the RandomForest model
        prediction_tod = rf_model_tod.predict(input_data_tod)[0]
        st.write(f"Predicted number of incidents: {prediction_tod}")
    else:
        st.error("No historical data available for the selected time of day and cluster.")
