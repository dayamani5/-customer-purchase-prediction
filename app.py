
import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

# Custom CSS
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    header {
    visibility: hidden;
}

.main {
    padding-top: 0rem;
}

.block-container {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] label {
    color: white !important;
}

    label {
        color: white !important;
        font-weight: bold;
    }

    h1 {
        color: #00FFD1;
        text-align: center;
        font-size: 50px;
    }

    div.stButton > button {
        background-color: #00FFD1;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 20px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #00c9a7;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] label {
        color: white !important;
        font-size: 18px;
        font-weight: bold;
    }

    section[data-testid="stSidebar"] .stSelectbox div {
        color: black;
    }

    </style>
    """,
    unsafe_allow_html=True
)
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Prediction", "Dashboard", "About"]
)
if menu == "Home":

    st.title("🛒 Customer Purchase Prediction")

    st.write("""
    Welcome to the Customer Purchase Prediction App.

    This machine learning application predicts whether a customer
    is likely to purchase a product based on customer behavior,
    shopping activity, and engagement patterns.
    """)

    st.image(
        "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d",
        use_container_width=True
    )
if menu == "Prediction":

    # Load Model
    model = joblib.load("customer_purchase_model.pkl")

    st.title("🛒 Customer Purchase Behavior Prediction")

    st.write(
        "Predict whether a customer will purchase a product based on shopping behavior."
    )

    # Inputs
    age = st.number_input("Age", 18, 65, 25)

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    annual_income = st.number_input(
        "Annual Income",
        20000,
        150000,
        50000
    )

    spending_score = st.number_input(
        "Spending Score",
        1,
        100,
        50
    )

    time_on_website = st.number_input(
        "Time on Website",
        1,
        60,
        20
    )

    previous_purchases = st.number_input(
        "Previous Purchases",
        0,
        20,
        5
    )

    product_category = st.selectbox(
        "Product Category",
        ["Beauty", "Clothing", "Electronics", "Groceries", "Sports"]
    )

    discount_used = st.selectbox(
        "Discount Used",
        ["No", "Yes"]
    )

    # Encoding
    gender_encoded = 1 if gender == "Male" else 0

    discount_encoded = 1 if discount_used == "Yes" else 0

    category_mapping = {
        "Beauty": 0,
        "Clothing": 1,
        "Electronics": 2,
        "Groceries": 3,
        "Sports": 4
    }

    category_encoded = category_mapping[product_category]

    # Input Data
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "AnnualIncome": [annual_income],
        "SpendingScore": [spending_score],
        "TimeOnWebsite": [time_on_website],
        "PreviousPurchases": [previous_purchases],
        "ProductCategory": [category_encoded],
        "DiscountUsed": [discount_encoded]
    })

    # Prediction
    if st.button("Predict Purchase"):

        prediction = model.predict(input_data)
        prediction_probability = model.predict_proba(input_data)

        purchase_probability = prediction_probability[0][1] * 100

        if prediction[0] == 1:
            st.success("✅ Customer is likely to purchase.")
            st.info(f"Purchase Probability: {purchase_probability:.2f}%")
        else:
            st.error("❌ Customer is not likely to purchase.")
            st.info(f"Purchase Probability: {purchase_probability:.2f}%")
if menu == "Dashboard":

   if menu == "Dashboard":

    st.title("📊 Customer Insights Dashboard")

    df = pd.read_csv("customer_purchase_behavior.csv")

    st.subheader("Purchase Distribution")

    purchase_counts = df["Purchased"].value_counts()

    st.write("0 = Not Purchased, 1 = Purchased")
    st.write(purchase_counts)

    st.subheader("Average Values")

    avg_income = df["AnnualIncome"].mean()
    avg_spending = df["SpendingScore"].mean()
    avg_time = df["TimeOnWebsite"].mean()

    st.write(f"Average Annual Income: {avg_income:.2f}")
    st.write(f"Average Spending Score: {avg_spending:.2f}")
    st.write(f"Average Time on Website: {avg_time:.2f}")

    st.subheader("Dataset Preview")

    st.write(df.head(10))
if menu == "About":

    st.title("ℹ️ About Project")

    st.write("""
    ## Customer Purchase Behavior Prediction

    This project predicts whether a customer is likely to purchase a product
    based on customer behavior and shopping activity.

    ### Technologies Used
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Joblib

    ### Machine Learning Model
    Random Forest Classifier

    ### Features Used
    - Age
    - Gender
    - Annual Income
    - Spending Score
    - Time on Website
    - Previous Purchases
    - Product Category
    - Discount Used

    ### Project Outcome
    This application helps businesses understand customer purchasing behavior
    and predict future purchases using machine learning.
    """)

    st.success("✅ Project Successfully Developed using Machine Learning and Streamlit")
