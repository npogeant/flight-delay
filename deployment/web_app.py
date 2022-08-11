import io
import ast
import time

import numpy as np
import requests
import streamlit as st
from PIL import Image

API_ENDPOINT = "http://127.0.0.1:3000/predict"

# Create the header page content
st.title("")
st.markdown(
    "<h1 style='text-align: center; color: white;'>Flight Delay Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: grey;'>Predict the delay of your next flight</h1>",
    unsafe_allow_html=True,
)

image = Image.open('src/flight.jpg')

st.markdown("""---""")
st.image(image)
st.markdown("""---""")

days = {
    'Monday': '1',
    'Tuesday': '2',
    'Wednesday': '3',
    'Thursday': '4',
    'Friday': '5',
    'Saturday': '6',
    'Sunday': '7',
}

# Build the form containing all the necessary inputs for the model
with st.form("my_form"):
    day_of_week = st.selectbox(
        'What day of the week is your flight scheduled ?',
        ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'),
    )
    day_of_week = days[day_of_week]

    with open("airport_&_airline_iata.txt", encoding="utf-8") as load_file:
        tuples = [tuple(line.split()) for line in load_file]
        airport_iata = tuples[0]
        airline_iata = tuples[1]

    op_carrier = st.selectbox(
        'What is the IATA code of the Airline Company of your flight ?', airline_iata
    )

    origin = st.selectbox(
        'What is the IATA code of the Airport from which the flight departs ?',
        airport_iata,
    )

    dep_time = st.text_input(
        'Insert the departure time (HHhmm in 24 hours format)', value='14h00'
    )
    dep_time = (
        dep_time[1:].replace('h', '')
        if dep_time[0] == '0'
        else dep_time.replace('h', '')
    )

    air_time = st.text_input('Insert the time of the flight (in minutes)', value='60')

    submitted = st.form_submit_button("Submit")


def predict(day_of_week, op_carrier, origin, dep_time, air_time):
    """
    A function that sends a prediction request to the API and return a cuteness score.

    Args:
        - day_of_week : 1st input
        - op_carrier : 2nd input
        - origin : 3rd input
        - dep_time : 4th input
        - air_time : 5th input
    """
    # Convert the bytes image to a NumPy array
    inputs = f'[[{day_of_week}, "{op_carrier}", "{origin}", {dep_time}, {air_time}]]'

    # Send the image to the API
    response = requests.post(API_ENDPOINT, data=inputs)

    if response.status_code == 200:  # pylint: disable=no-else-return
        return response.text
    else:
        raise Exception(f"Status: {response.status_code}")


# This function returns a success answer if everything goes right
def main():
    if submitted:
        with st.spinner("Predicting..."):
            time.sleep(2)
            try:
                prediction = predict(
                    day_of_week, op_carrier, origin, dep_time, air_time
                )
                prediction = ast.literal_eval(prediction)[0]
                if float(prediction) > 0.5:
                    st.success("Your flight will be on time 😎")
                elif float(prediction) <= 0.5:
                    st.success("Your flight will be delayed 🥶")
            except Exception:  # pylint: disable=broad-except
                st.warning('Something went wrong ... please check your inputs 😳')


if __name__ == "__main__":
    main()
