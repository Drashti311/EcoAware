import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_data
def load_data():
    df = pd.read_csv("Carbon Emission.csv")

    # Match notebook preprocessing
    df["Vehicle Type"] = df["Vehicle Type"].fillna("No Vehicle")

    q1 = df["CarbonEmission"].quantile(0.25)
    q3 = df["CarbonEmission"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df["CarbonEmission"] >= lower) & (df["CarbonEmission"] <= upper)]

    return df


@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model.pkl")


df = load_data()
model = load_model()


st.set_page_config(
    page_title="Full Carbon Emission Assistant",
    page_icon="🌍",
    layout="wide",
)

# Top layout
header_col, kpi_col2, kpi_col3 = st.columns([2, 1, 1])

st.title("🌍EcoAware - Carbon Emission Assistant")
st.write(
    "Fill in the details below to estimate your yearly carbon emissions and "
    "see where small lifestyle changes could have the biggest impact."
)

st.markdown("### Please Enter Your household and habits")

with st.form("carbon_form_full"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### Basics")
        body_type = st.selectbox("Body type", sorted(df["Body Type"].unique()))
        sex = st.selectbox("Sex", sorted(df["Sex"].unique()))
        diet = st.selectbox("Diet", sorted(df["Diet"].unique()))
        shower = st.selectbox(
            "How often do you shower?", df["How Often Shower"].unique()
        )
        heating = st.selectbox(
            "Main heating energy source", sorted(df["Heating Energy Source"].unique())
        )

    with col2:
        st.markdown("#### Transport & travel")
        transport = st.selectbox(
            "Main daily transport mode", sorted(df["Transport"].unique())
        )
        vehicle_type = st.selectbox(
            "Vehicle type", sorted(df["Vehicle Type"].unique())
        )
        vehicle_km = st.number_input(
            "Vehicle distance per month (km)",
            min_value=0,
            max_value=int(df["Vehicle Monthly Distance Km"].max()),
            value=int(df["Vehicle Monthly Distance Km"].median()),
        )
        air = st.selectbox(
            "How often do you travel by air?",
            df["Frequency of Traveling by Air"].unique(),
        )
        social = st.selectbox(
            "How often do you go out socially?", df["Social Activity"].unique()
        )

    with col3:
        st.markdown("#### Home, waste & media")

        grocery = st.number_input(
            "Monthly grocery bill (your currency)",
            min_value=0,
            max_value=5000,
            value=int(df["Monthly Grocery Bill"].median()),
            step=10,
        )
        waste_size = st.selectbox(
            "Typical waste bag size", df["Waste Bag Size"].unique()
        )
        waste_count = st.number_input(
            "Waste bags per week",
            min_value=0,
            max_value=int(df["Waste Bag Weekly Count"].max()),
            value=int(df["Waste Bag Weekly Count"].median()),
        )

        tv_hours = st.number_input(
            "TV / computer hours per day",
            min_value=0,
            max_value=int(df["How Long TV PC Daily Hour"].max()),
            value=int(df["How Long TV PC Daily Hour"].median()),
        )
        internet_hours = st.number_input(
            "Internet use hours per day",
            min_value=0,
            max_value=int(df["How Long Internet Daily Hour"].max()),
            value=int(df["How Long Internet Daily Hour"].median()),
        )

    with col4:
        st.markdown("#### Energy & recycling ")
        clothes = st.number_input(
            "New clothing items / month",
            min_value=0,
            max_value=int(df["How Many New Clothes Monthly"].max()),
            value=int(df["How Many New Clothes Monthly"].median()),
        )
        recycling = st.selectbox("Recycling habits", df["Recycling"].unique())

        efficiency = st.selectbox(
            "Home energy efficiency", df["Energy efficiency"].unique()
        )

        cooking_with = st.selectbox(
            "Main cooking appliances", df["Cooking_With"].unique()
        )

    submitted = st.form_submit_button("🔍 Calculate my footprint")

if submitted:
    user_row = {
        "Body Type": body_type,
        "Sex": sex,
        "Diet": diet,
        "How Often Shower": shower,
        "Heating Energy Source": heating,
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Social Activity": social,
        "Monthly Grocery Bill": grocery,
        "Frequency of Traveling by Air": air,
        "Vehicle Monthly Distance Km": vehicle_km,
        "Waste Bag Size": waste_size,
        "Waste Bag Weekly Count": waste_count,
        "How Long TV PC Daily Hour": tv_hours,
        "How Many New Clothes Monthly": clothes,
        "How Long Internet Daily Hour": internet_hours,
        "Energy efficiency": efficiency,
        "Recycling": recycling,
        "Cooking_With": cooking_with,
    }

    user_df = pd.DataFrame([user_row])
    pred = float(model.predict(user_df)[0])

    low_thr = df["CarbonEmission"].quantile(0.33)
    mid_thr = df["CarbonEmission"].quantile(0.66)

    if pred <= low_thr:
        level = "Low"
    elif pred <= mid_thr:
        level = "Moderate"
    else:
        level = "High"

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🌍 Your estimated emissions")
        st.metric("Predicted carbon emission", f"{pred:.2f} units")
        st.caption(f"Dataset average: {df['CarbonEmission'].mean():.2f} units")

    with col_b:
        st.subheader("📊 Emission level")
        st.metric("Category", level)
        percentile = (df["CarbonEmission"] < pred).mean() * 100
        st.caption(
            f"You emit more than roughly {percentile:.0f}% of people in this dataset."
        )

    st.markdown("### 📌 Personalised recommendations")

    tips = []

    if vehicle_km > df["Vehicle Monthly Distance Km"].median():
        if transport == "private":
            tips.append(
                "Try to reduce private car travel by combining trips, using public "
                "transport where possible, or walking/cycling for short distances."
            )
        elif transport == "public":
            tips.append(
                "You already use public transport often. Look for any remaining car "
                "trips and see if they can be shifted to transit, walking or cycling."
            )
        else:
            tips.append(
                "Your main mode is walking/cycling, which is great. For occasional "
                "car trips, consider sharing rides and combining errands."
            )

    if air in ["frequently", "very frequently"]:
        tips.append(
            "Consider cutting one flight per year or replacing short flights with "
            "train/bus travel when available."
        )

    if clothes > df["How Many New Clothes Monthly"].median():
        tips.append(
            "Reduce fast-fashion purchases; buy fewer, higher-quality items, and "
            "reuse or repair clothes where possible."
        )

    if waste_count > df["Waste Bag Weekly Count"].median():
        tips.append(
            "Improve waste sorting and recycling so that general rubbish shrinks to "
            "fewer, smaller bags per week."
        )

    if efficiency == "No":
        tips.append(
            "Look into basic home efficiency upgrades such as LED bulbs, better "
            "insulation and careful thermostat use."
        )

    if not tips:
        st.success(
            "Your lifestyle already looks relatively low-impact in this model. "
            "Keep focusing on low-carbon transport, efficient energy use and low waste."
        )
    else:
        for t in tips:
            st.markdown(f"- ✅ {t}")
