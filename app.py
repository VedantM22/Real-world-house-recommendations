import streamlit as st
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample user input (replace with your own)
data = pd.read_csv('clean.csv')

st.title("Property Recommendations")
st.sidebar.title("User Input")

# Create input fields for user preferences
pincode, bedrooms, transaction_type, property_type, carpet_area_sqft, price = None, None, None, None, None, None

pincode_options = ['Any'] + data['pincode'].unique().tolist()
pincode = st.sidebar.selectbox("Select Pincode", pincode_options)

bedrooms_options = ['Any'] + [str(i) for i in range(1, 11)]
bedrooms = st.sidebar.selectbox("Select Number of Bedrooms", bedrooms_options)

transaction_type_options = ['Any', 'New Property', 'Resale']
transaction_type = st.sidebar.selectbox("Select Transaction Type", transaction_type_options)

property_type_options = ['Any', 'Apartment', 'Villa', 'Penthouse']
property_type = st.sidebar.selectbox("Select Property Type", property_type_options)

carpet_area_sqft = st.sidebar.number_input("Enter Carpet Area (sqft)", 0, 10000)
price = st.sidebar.number_input("Enter Price (INR)", 0, 1000000000)

if st.sidebar.button("Recommend"):
    # Filter data based on user input for pincode and apartment type
    filtered_data = data.copy()
    
    if pincode != 'Any':
        filtered_data = filtered_data[filtered_data['pincode'] == int(pincode)]
    
    if property_type != 'Any':
        filtered_data = filtered_data[filtered_data['property_type'] == property_type]

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    filtered_data['combined_features'] = filtered_data[['pincode', 'bedrooms', 'transaction_type', 'property_type', 'carpet_area_sqft', 'price']].apply(lambda row: ' '.join(map(str, row)), axis=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['combined_features'])

    user_input_str = ' '.join(map(str, [pincode, bedrooms, transaction_type, property_type, carpet_area_sqft, price]))
    user_tfidf = tfidf_vectorizer.transform([user_input_str])

    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    sqft_weight = 1.0  # Adjust the weight for carpet area (sqft)
    price_weight = 2.0  # Adjust the weight for price

    # Check if 'carpet_area_sqft' and 'price' exist in the vocabulary
    if 'carpet_area_sqft' in tfidf_vectorizer.vocabulary_ and 'price' in tfidf_vectorizer.vocabulary_:
        filtered_data['similarity'] = cosine_similarities
        filtered_data['similarity'] += sqft_weight * tfidf_matrix[:, tfidf_vectorizer.vocabulary_['carpet_area_sqft']].toarray().flatten()
        filtered_data['similarity'] += price_weight * tfidf_matrix[:, tfidf_vectorizer.vocabulary_['price']].toarray().flatten()
    else:
        filtered_data['similarity'] = cosine_similarities

    sorted_data = filtered_data.sort_values(by='similarity', ascending=False)

    N = 5

    st.subheader("Recommendations for the user:")
    recommendations = sorted_data.head(N)

    # Iterate through the recommendations and create clickable links
    for index, row in recommendations.iterrows():
        st.markdown(f"[{row['location']} - {row['property_type']} - {row['pincode']} - {row['carpet_area_sqft']} sqft - {row['bedrooms']} BHK - ₹{row['price']}]({row['links']})")
