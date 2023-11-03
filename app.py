import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load your dataset
df = pd.read_csv('FINAL_cap.csv')

# Set page layout and style
st.set_page_config(
    page_title="Property Recommendations",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Property Recommendations")
st.sidebar.title("User Input")

location_with_pincode = st.sidebar.selectbox('Select the location', df['location_pincode'].unique(), index=0)
pincode = int(location_with_pincode.split('-')[1])

bedroom_options = sorted(df[df['pincode'] == pincode]['bed'].unique().tolist())
bedroom = st.sidebar.selectbox('Bedrooms', bedroom_options, index=None)

furniture_options = sorted(df[df['pincode'] == pincode]['furnished status'].unique().tolist())
furniture_opt = [value for value in furniture_options if value != 'Not Mentioned']
furniture = st.sidebar.selectbox('Furniture type', furniture_opt, index=None)

#facing_options = sorted(df[df['pincode'] == pincode]['facing'].unique().tolist())
#facing = st.sidebar.selectbox('Facing', facing_options, index=None)

price = st.sidebar.text_input('Enter the price')

rank=df[df['pincode']== pincode]['rank'].iloc[0]

# Create a custom TF-IDF vectorizer with column-specific weights
df['feature_vector'] = df[['pincode', 'prc', 'bed', 'furnished status', 'rank']].apply(lambda x: ' '.join(map(str, x)), axis=1)

tfidf_weights = {
    'prc': 1,
    'pincode': 1,
    'bed': 1,
    'furnished status': 0.70,
    'rank': 1,  # Adjust the weight for the 'rank' feature
}

tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False, stop_words=None, use_idf=False, norm=None, smooth_idf=False)
tfidf_vectorizer.fit(df['feature_vector'])

tfidf_matrix = tfidf_vectorizer.transform(df['feature_vector'])
for col in tfidf_weights:
    col_idx = tfidf_vectorizer.vocabulary_.get(col)
    if col_idx is not None:
        tfidf_matrix[:, col_idx] = tfidf_matrix[:, col_idx] * tfidf_weights[col]


# Define your recommendation function
def recommend_properties(df, pincode, price, bedroom, furniture, rank):
    user_input = f"{pincode} {price} {bedroom} {furniture} {rank}"

    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    cosine_similarities = linear_kernel(user_input_tfidf, tfidf_matrix)

    n_recommendations = 30
    similar_indices = cosine_similarities[0].argsort()[:-n_recommendations-1:-1]

    recommended_properties = df.iloc[similar_indices][['prc', 'bed', 'web', 'pincode','amenities','carpet area', 'location_pincode','location', 'facing', 'furnished status', 'link', 'status', 'price']]
    recommended_properties = recommended_properties.sort_values(by='prc')

    return recommended_properties

if st.sidebar.button("Recommend"):
    if pincode:
        recommended_properties = recommend_properties(df, pincode, price, bedroom, furniture, rank)

        if not recommended_properties.empty:
            st.header("Here are the recommendations")

            for i, row in recommended_properties.head(20).iterrows():
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Website:** {row['web']}")
                    st.markdown(f"**Price:** ₹{row['price']}")
                    st.markdown(f"**Location:** {row['location_pincode']}")
                    st.markdown(f"**Bedrooms:** {row['bed']}")

                with col2:
                    st.markdown(f"**Sqft:** {row['carpet area']}")
                    st.markdown(f"**Furnished Status:** {row['furnished status']}")
                    st.markdown(f"**Possession:** {row['status']}")
                    st.markdown(f"**Facing:** {row['facing']}")
                    property_link = row['link']
                    st.markdown(f"[View more details]({property_link})")
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("No properties found for the specified pincode.")
    else:
        st.warning("Please enter a pincode to get recommendations.")
