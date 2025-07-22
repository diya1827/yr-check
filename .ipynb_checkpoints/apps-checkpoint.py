import pandas as pd
import cloudpickle
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
from sklearn.compose import ColumnTransformer

# @st.cache_resource ensures the model and encoder are loaded only once
@st.cache_resource
def load_model_and_encoder():
    try:
        with open("model.pkl", "rb") as f:
            model = cloudpickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoder = cloudpickle.load(f)
        return model, encoder
    except FileNotFoundError:
        st.error("Error: Model or encoder files (model.pkl, encoder.pkl) not found. Please ensure they are in the same directory as the script.")
        st.stop() # Stop the app if files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model or encoder: {e}")
        st.stop()

def main():
    # Load model and encoder using the cached function
    model, encoder = load_model_and_encoder()

    # Define options
    hunger_options = ['Somewhat Hungry', 'Very Hungry']
    meal_options = ['Breakfast', 'Lunch', 'Evening Refreshments', 'Dinner']
    budget_options = ['<50', '5-100', '100-150', '150-200', '200-500', '500-1000', '1000+']
    drink_options = ['Yes', 'No']
    food_options = ['Veg', 'Non-Veg']
    travel_options = ['<5 min', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']

    # Place label map
    place_label_map = {
        6: 'hungry yak/dtu cafe',
        8: 'yellow bowl',
        4: 'baozi',
        3: 'Rohini Market cafes',
        9: 'zomato/swiggy/online mode',
        1: 'Mcd, KFC other fast food outlets',
        10: 'Bunker house/Yogit Mess/Apsara',
        7: 'street food from front of the campus',
        5: 'crazy crisp',
        2: 'Dillicious/Madras Cafe'
    }

    st.title("YUMRADAR: OFF CAMPUS RECOMMENDATIONS")

    with st.form("user_input_form"):
        hunger_level = st.selectbox("1. Hunger Level", hunger_options)
        meal_type = st.selectbox("2. Meal Type", meal_options)
        budget = st.selectbox("3. Budget Range (₹)", budget_options)
        drinks = st.selectbox("4. Prefer Drinks?", drink_options)
        food_type = st.selectbox("5. Food Preference", food_options)
        travel_time = st.selectbox("6. Travel Time", travel_options)

        submit = st.form_submit_button("Predict Place")

    if submit:
        # Map categorical inputs to numerical values
        hunger_map = {'Somewhat Hungry': 0, 'Very Hungry': 1}
        meal_map = {'Breakfast': 0, 'Lunch': 1, 'Evening Refreshments': 2, 'Dinner': 3}
        drink_map = {'No': 0, 'Yes': 1}
        food_map = {'Veg': 0, 'Non-Veg': 1}

        # Create a DataFrame for the input, ensuring column names match training data
        input_df = pd.DataFrame([{
            'hunger_level_': hunger_map[hunger_level],
            'frequent_meal_type_': meal_map[meal_type],
            'budget_range': budget,
            'prefer_drinks_': drink_map[drinks],
            'prefer_food_type_': food_map[food_type],
            'travel_time_range': travel_time
        }])

        try:
            # Transform the input using the loaded encoder
            input_encoded = encoder.transform(input_df)

            # Make a prediction
            prediction = model.predict(input_encoded)

            predicted_label = prediction[0]
            place_name = place_label_map.get(predicted_label, "Unknown Place")

            st.success(f"📍 You should visit : **{place_name}**")
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs or the model/encoder: {e}")


if __name__ == "__main__":
    main()
