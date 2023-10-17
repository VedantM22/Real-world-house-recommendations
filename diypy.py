

from lib2to3.pgen2.pgen import DFAState
import streamlit as st
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
#import regex as re

# Sample user input (replace with your own)
df = pd.read_csv('clean.csv')

# Set page layout and style
st.set_page_config(
    page_title="Property Recommendations",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Property Recommendations")
st.sidebar.title("User Input")

# Create input field for location selection
location_with_pincode = st.sidebar.selectbox('Select the lcoation' , df['location_with_pincode'].unique().tolist(),index=None)
bedroom_options =  df[df['location_with_pincode']==location_with_pincode]['bedrooms'].unique()
bedroom = st.sidebar.selectbox('Bedrooms',['Any']+ bedroom_options.tolist(),index=None)

property_options = df[df['location_with_pincode']==location_with_pincode]['property_type'].unique()
property = st.sidebar.selectbox('Transaction type',['Any']+ property_options.tolist(),index=None)

transaction_options = df[df['location_with_pincode']==location_with_pincode]['transaction_type'].unique()
transaction = st.sidebar.selectbox('Transaction type',['Any']+ ['New Property','Resale'],index=None)

price = st.sidebar.number_input('Enter your maximum budget',value=0)



if st.sidebar.button("Recommend"):
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    df['combined_features'] = df[['location_with_pincode', 'bedrooms', 'transaction_type', 'property_type', 'price']].apply(lambda row: ' '.join(map(str, row)), axis=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    user_input_str = ' '.join(map(str, [location_with_pincode, bedroom, transaction, property, price]))
    user_tfidf = tfidf_vectorizer.transform([user_input_str])

    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()
    
    df['similarity'] = cosine_similarities
    sorted_data = df.sort_values(by='similarity', ascending=False)



    filtered_data = df[
        (df['location_with_pincode'] == location_with_pincode) &
        (bedroom == 'Any' or df['bedrooms'] == bedroom) &
        (transaction == 'Any' or df['transaction_type'] == transaction) &
        (property == 'Any' or df['property_type'] == property)
    ]

    filter_data = filtered_data[filtered_data['price'] <= price].head(10)

    if len(filter_data) < 10:
        error = 10 - int(len(filter_data))
        st.header(f"There are {len(filter_data)} such apartments available, So here are the those apartments and some other recommendations")
        sorted_data_new = sorted_data.head(error) 
        rec_data = pd.concat([filter_data, sorted_data_new], axis=0)
        col1, col2 = st.columns(2)
        for i, row in rec_data.iterrows():
            formatted_price = "{:.2f}".format(row['price'] / 10000000)
            with col1:
                st.markdown(f"**Location with Pincode:** {row['location_with_pincode']}")
                st.markdown(f"**Bedrooms:** {row['bedrooms']}")
                st.markdown(f"**Property Type:** {row['property_type']}")
                st.markdown(f"**Transaction Type:** {row['transaction_type']}")
                st.markdown("---")
            with col2:
                st.markdown(f"**Price:** ₹{formatted_price} crores ({int(row['price'])} INR)")
                st.markdown(f"**Furniture:** {row['furniture']}")
                st.markdown(f"**Parking:** {row['parking']}")
                property_link = row['links']
                st.markdown(f"[Click here]({property_link})")
                st.markdown("---")
    elif len(filter_data) == 10:
        st.header("Here are the recommendations")
        col1, col2 = st.columns(2)
        for i, row in filter_data.iterrows():
            formatted_price = "{:.2f}".format(row['price'] / 10000000)
            with col1:
                st.markdown(f"**Location with Pincode:** {row['location_with_pincode']}")
                st.markdown(f"**Bedrooms:** {row['bedrooms']}")
                st.markdown(f"**Property Type:** {row['property_type']}")
                st.markdown(f"**Transaction Type:** {row['transaction_type']}")
                st.markdown("---")
            with col2:
                st.markdown(f"**Price:** ₹{formatted_price} crores ({int(row['price'])} INR)")
                st.markdown(f"**Furniture:** {row['furniture']}")
                st.markdown(f"**Parking:** {row['parking']}")
                property_link = row['links']
                st.markdown(f"[Click here]({property_link})")
                st.markdown("---")
