import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="APL Logistics - Late Delivery Risk Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #F4F6F9;
}

[data-testid="stSidebar"] {
    background-color: #0F172A;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: #0F172A;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:12px;
    border:1px solid #E5E7EB;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        "APL_Logistics.csv",
        encoding="latin1"
    )
    return df

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "Late_Delivery_Risk_Model.pkl"
    )

    scaler = joblib.load(
        "Scaler.pkl"
    )

    return model, scaler

# ==========================================================
# LOAD LABEL ENCODERS
# ==========================================================

@st.cache_resource
def load_encoders():

    return joblib.load(
        "LabelEncoders.pkl"
    )

# ==========================================================
# LOAD FEATURE LIST
# ==========================================================

@st.cache_resource
def load_feature_names():

    return joblib.load(
        "model_features.pkl"
    )

# ==========================================================
# LOAD EVERYTHING
# ==========================================================

df = load_data()

model, scaler = load_model()

encoders = load_encoders()

model_features = load_feature_names()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📦 APL Logistics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Risk Analytics",
        "Regional Analytics",
        "Prediction Center",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Late Delivery Risk Prediction\n\nMachine Learning Dashboard"
)


# ==========================================================
# TEST PAGE
# ==========================================================

# ==========================================================
# EXECUTIVE DASHBOARD
# ==========================================================

if page == "Executive Dashboard":

    st.title("APL Logistics - Executive Dashboard")

    st.markdown("---")

    # KPI VALUES

    total_orders = len(df)

    late_orders = df["Late_delivery_risk"].sum()

    late_percentage = round(
        (late_orders / total_orders) * 100,
        2
    )

    total_sales = round(
        df["Sales"].sum(),
        2
    )

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Late Deliveries",
        f"{late_orders:,}"
    )

    col3.metric(
        "Late Delivery %",
        f"{late_percentage}%"
    )

    col4.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    st.markdown("---")

    left, right = st.columns(2)

    # --------------------------------------------------
    # Late Delivery Distribution
    # --------------------------------------------------

    with left:

        pie = px.pie(

            df,

            names="Late_delivery_risk",

            title="Late Delivery Distribution"

        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    # --------------------------------------------------
    # Shipping Mode
    # --------------------------------------------------

    with right:

        shipping = df["Shipping Mode"].value_counts().reset_index()

        shipping.columns = [
            "Shipping Mode",
            "Orders"
        ]

        fig = px.bar(

            shipping,

            x="Shipping Mode",

            y="Orders",

            color="Orders",

            title="Orders by Shipping Mode"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    left2, right2 = st.columns(2)

    # --------------------------------------------------
    # Market Analysis
    # --------------------------------------------------

    with left2:

        market = df["Market"].value_counts().reset_index()

        market.columns = [
            "Market",
            "Orders"
        ]

        fig = px.bar(

            market,

            x="Market",

            y="Orders",

            color="Orders",

            title="Orders by Market"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Order Status
    # --------------------------------------------------

    with right2:

        status = df["Order Status"].value_counts().reset_index()

        status.columns = [
            "Order Status",
            "Orders"
        ]

        fig = px.bar(

            status,

            x="Order Status",

            y="Orders",

            color="Orders",

            title="Order Status Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    category = (
        df["Category Name"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    category.columns = [
        "Category",
        "Orders"
    ]

    fig = px.bar(

        category,

        x="Orders",

        y="Category",

        orientation="h",

        color="Orders",

        title="Top 10 Product Categories"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# RISK ANALYTICS
# ==========================================================

elif page == "Risk Analytics":

    st.title("Risk Analytics")

    st.markdown("---")

    high_risk = len(df[df["Late_delivery_risk"] == 1])

    low_risk = len(df[df["Late_delivery_risk"] == 0])

    risk_percentage = round(
        (high_risk / len(df)) * 100,
        2
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High Risk Orders",
        f"{high_risk:,}"
    )

    col2.metric(
        "Low Risk Orders",
        f"{low_risk:,}"
    )

    col3.metric(
        "High Risk %",
        f"{risk_percentage}%"
    )

    st.markdown("---")

    left, right = st.columns(2)

    # --------------------------------------------------
    # Late Delivery Risk Distribution
    # --------------------------------------------------

    with left:

        fig = px.pie(
            df,
            names="Late_delivery_risk",
            title="Late Delivery Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Risk by Shipping Mode
    # --------------------------------------------------

    with right:

        shipping_risk = (
            df.groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        shipping_risk["Late_delivery_risk"] *= 100

        fig = px.bar(
            shipping_risk,
            x="Shipping Mode",
            y="Late_delivery_risk",
            color="Late_delivery_risk",
            title="Average Risk by Shipping Mode"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    left2, right2 = st.columns(2)

    # --------------------------------------------------
    # Market Risk
    # --------------------------------------------------

    with left2:

        market_risk = (
            df.groupby("Market")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        market_risk["Late_delivery_risk"] *= 100

        fig = px.bar(
            market_risk,
            x="Market",
            y="Late_delivery_risk",
            color="Late_delivery_risk",
            title="Average Risk by Market"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Order Status Distribution
    # --------------------------------------------------

    with right2:

        status = (
            df.groupby("Order Status")["Late_delivery_risk"]
            .count()
            .reset_index()
        )

        fig = px.bar(
            status,
            x="Order Status",
            y="Late_delivery_risk",
            color="Late_delivery_risk",
            title="Orders by Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    category = (
        df.groupby("Category Name")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    category["Late_delivery_risk"] *= 100

    fig = px.bar(
        category,
        x="Late_delivery_risk",
        y="Category Name",
        orientation="h",
        color="Late_delivery_risk",
        title="Top 10 Highest Risk Categories"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# REGIONAL ANALYTICS
# ==========================================================

elif page == "Regional Analytics":

    st.title("Regional Analytics")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Markets",
        df["Market"].nunique()
    )

    col2.metric(
        "Countries",
        df["Order Country"].nunique()
    )

    col3.metric(
        "States",
        df["Order State"].nunique()
    )

    st.markdown("---")

    left, right = st.columns(2)

    # --------------------------------------------------
    # Orders by Market
    # --------------------------------------------------

    with left:

        market = (
            df.groupby("Market")
            .size()
            .reset_index(name="Orders")
        )

        fig = px.bar(
            market,
            x="Market",
            y="Orders",
            color="Orders",
            title="Orders by Market"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Orders by Country
    # --------------------------------------------------

    with right:

        country = (
            df.groupby("Order Country")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="Orders")
        )

        fig = px.bar(
            country,
            x="Order Country",
            y="Orders",
            color="Orders",
            title="Top 10 Countries by Orders"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    left2, right2 = st.columns(2)

    # --------------------------------------------------
    # Orders by Region
    # --------------------------------------------------

    with left2:

        region = (
            df.groupby("Order Region")
            .size()
            .reset_index(name="Orders")
        )

        fig = px.bar(
            region,
            x="Order Region",
            y="Orders",
            color="Orders",
            title="Orders by Region"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Top States
    # --------------------------------------------------

    with right2:

        state = (
            df.groupby("Order State")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="Orders")
        )

        fig = px.bar(
            state,
            x="Order State",
            y="Orders",
            color="Orders",
            title="Top 10 States by Orders"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    market_sales = (
        df.groupby("Market")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        market_sales,
        names="Market",
        values="Sales",
        title="Sales Contribution by Market"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# PREDICTION CENTER
# ==========================================================

elif page == "Prediction Center":

    st.title("Late Delivery Prediction")

    st.markdown("---")

    st.write("Select an order from the dataset to predict its late delivery risk.")

    order_number = st.selectbox(

        "Select Order",

        df.index

    )

    selected_order = df.loc[[order_number]].copy()

    st.subheader("Selected Order")

    st.dataframe(selected_order)

    if st.button("Predict Late Delivery"):

        prediction_data = selected_order.copy()

        prediction_data = prediction_data.drop(
            columns=["Late_delivery_risk"],
            errors="ignore"
        )

        # Encode categorical columns
        for col, encoder in encoders.items():

            if col in prediction_data.columns:

                prediction_data[col] = prediction_data[col].astype(str)

                prediction_data[col] = prediction_data[col].apply(
                    lambda x: encoder.transform([x])[0]
                    if x in encoder.classes_
                    else -1
                )

        # Keep only features used during training
        prediction_data = prediction_data.reindex(
            columns=model_features,
            fill_value=0
        )

        # Scale
        prediction_scaled = scaler.transform(prediction_data)

        # Predict
        prediction = model.predict(prediction_scaled)[0]

        probability = model.predict_proba(
            prediction_scaled
        )[0][1]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("Late Delivery Risk: HIGH")

        else:

            st.success("Late Delivery Risk: LOW")

        st.metric(
            "Probability of Late Delivery",
            f"{probability*100:.2f}%"
        )

        if probability >= 0.80:

            st.warning(
                "Recommendation: Immediate operational attention required."
            )

        elif probability >= 0.50:

            st.info(
                "Recommendation: Monitor this shipment carefully."
            )

        else:

            st.success(
                "Recommendation: Shipment appears to be on schedule."
            )

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.markdown("---")

    st.subheader("Model Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Algorithm", "XGBoost")
    c2.metric("Target", "Late Delivery Risk")
    c3.metric("Problem Type", "Classification")
    c4.metric("Features", len(model_features))

    st.markdown("---")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(df.columns))
    col3.metric("Late Deliveries", int(df["Late_delivery_risk"].sum()))

    st.markdown("---")

    st.subheader("Late Delivery Distribution")

    fig = px.pie(
        df,
        names="Late_delivery_risk",
        title="Target Variable Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Feature Importance")

    try:

        importance = model.feature_importances_

        importance_df = pd.DataFrame({

            "Feature": model_features,
            "Importance": importance

        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(15)

        fig = px.bar(

            importance_df,

            x="Importance",

            y="Feature",

            orientation="h",

            title="Top 15 Important Features"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception:

        st.warning("Feature importance is not available for this model.")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.title("About Project")

    st.markdown("---")

    st.header("APL Logistics - Late Delivery Risk Prediction")

    st.write("""
This project uses **Machine Learning (XGBoost Classifier)** to predict
whether an order is likely to experience a late delivery.

The dashboard provides interactive visualizations, business insights,
and a prediction interface to assist logistics managers in monitoring
shipment performance and reducing delivery delays.
""")

    st.markdown("---")

    st.subheader("Project Objectives")

    st.markdown("""
- Predict late delivery risk using Machine Learning.
- Analyze delivery performance across different markets.
- Identify high-risk shipments.
- Visualize logistics data using interactive dashboards.
- Support better operational decision-making.
""")

    st.markdown("---")

    st.subheader("Technology Stack")

    tech = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Plotly",
            "Scikit-Learn",
            "XGBoost",
            "Joblib"
        ],
        "Purpose": [
            "Programming Language",
            "Dashboard Development",
            "Data Analysis",
            "Numerical Computing",
            "Interactive Visualization",
            "Machine Learning Utilities",
            "Prediction Model",
            "Model Serialization"
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Records", len(df))
    c2.metric("Features", len(df.columns))
    c3.metric("Target", "Late Delivery Risk")

    st.markdown("---")

    st.success("Project Completed Successfully ")

    st.caption("Developed using Streamlit and XGBoost for Academic Purposes.")