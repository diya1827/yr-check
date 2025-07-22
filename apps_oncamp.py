import pandas as pd
import cloudpickle
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
from sklearn.compose import ColumnTransformer

# @st.cache_resource ensures the model and encoder are loaded only once
@st.cache_resource
def load_model_and_encoder():
    try:
        with open("model_oncampus.pkl", "rb") as f:
            model = cloudpickle.load(f)
        with open("encoder_oncampus.pkl", "rb") as f:
            encoder = cloudpickle.load(f)
        return model, encoder
    except FileNotFoundError:
        st.error("Error: Model or encoder files (model_oncampus.pkl, encoder_oncampus.pkl) not found. Please ensure they are in the same directory as the script.")
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

    # 🔁 Add label-to-name mapping (replace with your actual data)
    place_label_map = {
        5: 'Raj Soin',
        2: 'Hims',
        6: 'Raydee',
        1: 'Dosa Plaza',
        4: 'Nescafe',
        3: 'Mess',
        8: 'Deltech',
        9: 'Udupi',
        7: 'Bistro57'
        # Add more if needed
    }

    st.title("YUMRADAR: ON CAMPUS RECOMMENDATIONS")

    # Input form
    with st.form("user_input_form"):
        hunger_level = st.selectbox("1. Hunger Level", hunger_options)
        meal_type = st.selectbox("2. Meal Type", meal_options)
        budget = st.selectbox("3. Budget Range (₹)", budget_options)
        drinks = st.selectbox("4. Prefer Drinks?", drink_options)
        food_type = st.selectbox("5. Food Preference", food_options)
    

        submit = st.form_submit_button("Predict Place")

    # Handle prediction
    if submit:
         # Map options to integers (must match training data)
        hunger_map = {'Somewhat Hungry': 0, 'Very Hungry': 1}
        meal_map = {'Breakfast': 0, 'Lunch': 1, 'Evening Refreshments': 2, 'Dinner': 3}
        drink_map = {'No': 0, 'Yes': 1}
        food_map = {'Veg': 0, 'Non-Veg': 1}

        # Replace user input with numerical values
        input_df = pd.DataFrame([{
            'hunger_level_': hunger_map[hunger_level],
            'frequent_meal_type_': meal_map[meal_type],
            'budget_range': budget,
            'prefer_drinks_': drink_map[drinks],
            'prefer_food_type_': food_map[food_type],
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
