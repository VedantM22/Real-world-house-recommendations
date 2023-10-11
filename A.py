import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample user input (replace with your own)
data = pd.read_csv('clean.csv')

# Set page layout and style
st.set_page_config(
    page_title="Property Recommendations",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Property Recommendations")
st.sidebar.title("User Input")

# Create input fields for user preferences
bedrooms, transaction_type, property_type, price, location_with_pincode, sqft = None, None, None, None, None, None

location_options = ['Any'] + data['location_with_pincode'].unique().tolist()
location_with_pincode = st.sidebar.selectbox("Select Location with Pincode", location_options)

if location_with_pincode != 'Any':
    # Fetch latitude and longitude from the dataset based on the selected location_with_pincode
    selected_property = data[data['location_with_pincode'] == location_with_pincode].iloc[0]
    latitude = selected_property['Latitude']
    longitude = selected_property['Longitude']

    # Define a function to calculate minimum sqft and price based on the selected location_with_pincode
    def calculate_min_values(location_with_pincode):
        # Initialize minimum values to high values
        min_sqft = 999999
        min_price = 999999999

        # Iterate through the data to find the actual minimum values
        for _, row in data.iterrows():
            if row['location_with_pincode'] == location_with_pincode:
                min_sqft = min(min_sqft, row['carpet_area_sqft'])
                min_price = min(min_price, row['price'])

        return min_sqft, min_price

    # Get the minimum values based on the selected location_with_pincode
    min_sqft, min_price = calculate_min_values(location_with_pincode)

    # Display other input sections with calculated minimum values
    bedrooms_options = ['Any'] + [str(i) for i in range(1, 11)]
    bedrooms = st.sidebar.selectbox("Select Number of Bedrooms", bedrooms_options)
    transaction_type_options = ['Any', 'New Property', 'Resale']
    transaction_type = st.sidebar.selectbox("Select Transaction Type", transaction_type_options)
    property_type_options = ['Any', 'Apartment', 'Villa', 'Penthouse']
    property_type = st.sidebar.selectbox("Select Property Type", property_type_options)
    sqft = st.sidebar.number_input("Enter Carpet Area (sqft)", min_value=min_sqft, value=min_sqft, max_value=10000)
    price = st.sidebar.number_input("Enter Price (INR)", min_value=min_price, value=min_price, max_value=1000000000)

if st.sidebar.button("Recommend"):
    # Create a custom TF-IDF vectorizer that assigns weights to latitude and longitude
    def custom_tfidf_vectorizer(data):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        
        # Add weight to latitude and longitude
        latitude_weight = 2.0
        longitude_weight = 2.0
        
        # Find the indices of latitude and longitude in the vocabulary
        latitude_index = tfidf_vectorizer.vocabulary_['Latitude']
        longitude_index = tfidf_vectorizer.vocabulary_['Longitude']
        
        # Apply weights to latitude and longitude
        tfidf_matrix[:, latitude_index] *= latitude_weight
        tfidf_matrix[:, longitude_index] *= longitude_weight
        
        return tfidf_matrix

    # Combine features for TF-IDF
    data['combined_features'] = data[['bedrooms', 'transaction_type', 'property_type', 'price', 'location_with_pincode', 'carpet_area_sqft', 'Latitude', 'Longitude']].apply(lambda row: ' '.join(map(str, row)), axis=1)
    tfidf_matrix = custom_tfidf_vectorizer(data['combined_features'])

    user_input_str = ' '.join(map(str, [bedrooms, transaction_type, property_type, price, location_with_pincode, sqft, latitude, longitude]))
    user_tfidf = custom_tfidf_vectorizer([user_input_str])

    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    sorted_indices = cosine_similarities.argsort()[::-1]  # Sort in descending order

    N = 10  # Number of recommendations to show

    st.subheader("Top 10 Recommended Properties:")
    
    recommendations_found = False  # Initialize a flag to check if any recommendations are found

    for i in range(N):
        index = sorted_indices[i]
        property_info = data.iloc[index]
        
        # Calculate the price in crores with two decimal places
        price_in_crores = property_info['price'] / 10000000
        formatted_price = "{:.2f}".format(price_in_crores)

        # Display property details side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Location with Pincode:** {property_info['location_with_pincode']}")
            st.markdown(f"**Bedrooms:** {property_info['bedrooms']}")
            st.markdown(f"**Carpet Area (in sqft):** {property_info['carpet_area_sqft']}")
            st.markdown(f"**Property Type:** {property_info['property_type']}")
            st.markdown(f"**Latitude:** {property_info['Latitude']}")
            st.markdown(f"**Longitude:** {property_info['Longitude']}")
        
        with col2:
            st.markdown(f"**Transaction Type:** {property_info['transaction_type']}")
            st.markdown(f"**Price:** ₹{formatted_price} crores ({int(property_info['price'])} INR)")
            st.markdown(f"**Furniture:** {property_info['furniture']}")
            st.markdown(f"**Parking:** {property_info['parking']}")

        # Create a clickable URL for each listing
        property_link = property_info['links']
        st.markdown(f"[Click here]({property_link})")

        # Set the flag to indicate recommendations are found
        recommendations_found = True

        # Add a delimiter line
        st.markdown("---")
    
    # Check if no recommendations were found and display a message
    if not recommendations_found:
        st.info("Here are some of the similar results.")
