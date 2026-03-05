import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="EcoAware",
    page_icon="🌍",
    layout="wide",
)

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_carbon_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model.pkl")


df = load_data()
model = load_model()

st.title("🌍 EcoAware - Carbon Footprint Estimator")

st.caption(
    "Estimate your carbon footprint based on everyday habits like travel, diet, and energy use"
)

st.write("")

with st.form("carbon_form"):
    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 1])

    # Travel & Transport 
    with col1:

        st.markdown("### 🚗 Travel & Transport")
        st.caption("Your mobility and travel habits")

        transport = st.selectbox(
            "Main daily transport",
            sorted(df["Transport"].unique())
        )

        vehicle_type = st.selectbox(
            "Vehicle fuel type",
            ["Not sure"] + sorted(df["Vehicle Type"].unique())
        )

        vehicle_km = st.number_input(
            "Vehicle distance per month (km)",
            min_value=0,
            max_value=int(df["Vehicle Monthly Distance Km"].max()),
            value=int(df["Vehicle Monthly Distance Km"].median()),
        )

        air = st.selectbox(
            "Air travel frequency",
            sorted(df["Frequency of Traveling by Air"].unique())
        )

    # Home & Energy 
    with col2:
        st.markdown("### 🏠 Home & Energy")
        st.caption("Household energy and waste habits")

        waste_count = st.number_input(
            "Garbage bags per week",
            min_value=0,
            max_value=int(df["Waste Bag Weekly Count"].max()),
            value=int(df["Waste Bag Weekly Count"].median()),
        )

        waste_size = st.selectbox(
            "Garbage bag size",
            sorted(df["Waste Bag Size"].unique())
        )        

        heating = st.selectbox(
            "Heating energy source",
            sorted(df["Heating Energy Source"].unique())
        )
        recycling = st.selectbox(
                "Recycling habits",
                ["Not sure"] + sorted(df["Recycling"].unique())
        )

    # Consumption 
    with col3:
        st.markdown("### 🛒 Consumption & Lifestyle")
        st.caption("Daily consumption and purchasing habits")

        diet = st.selectbox(
            "Diet type",
            sorted(df["Diet"].unique())
        )

        cooking_with = st.selectbox(
            "Cooking method",
            sorted(df["Cooking_With"].unique())
        )

        grocery = st.number_input(
            "Monthly grocery spending ($)",
            min_value=0,
            max_value=int(df["Monthly Grocery Bill"].max()),
            value=int(df["Monthly Grocery Bill"].median()),
        )

        clothes = st.number_input(
            "Clothing items per month",
            min_value=0,
            max_value=int(df["How Many New Clothes Monthly"].max()),
            value=int(df["How Many New Clothes Monthly"].median()),
        )

    # Additional Info 
    with col4:
        st.markdown("### 👤 Additional Info")
        st.caption("Optional details to improve prediction")

        with st.expander("Add additional information"):

            sex = st.selectbox(
                "Gender",
                ["Prefer not to say"] + sorted(df["Sex"].unique())
            )

            body_type = st.selectbox(
                "Body type",
                ["Prefer not to say"] + sorted(df["Body Type"].unique())
            )

            social = st.selectbox(
                "Social activity",
                ["Not sure"] + sorted(df["Social Activity"].unique())
            )
            efficiency = st.selectbox(
            "Energy efficiency",
            ["Not sure"] + sorted(df["Energy efficiency"].unique())
            )

            shower = st.selectbox(
                "Shower frequency",
                sorted(df["How Often Shower"].unique())
            )

            tv_hours = st.number_input(
                "TV / computer hours per day",
                min_value=0,
                max_value=int(df["How Long TV PC Daily Hour"].max()),
                value=int(df["How Long TV PC Daily Hour"].median()),
            )

            internet_hours = st.number_input(
                "Internet hours per day",
                min_value=0,
                max_value=int(df["How Long Internet Daily Hour"].max()),
                value=int(df["How Long Internet Daily Hour"].median()),
            )
            
    st.write("")
    st.write("")

    col_submit1, col_submit2, col_submit3 = st.columns([2, 1, 2])
    with col_submit2:
        submitted = st.form_submit_button("🔍 Calculate My Carbon Footprint")

# Prediction 

if submitted:
    default_row = {}

    for col in df.columns:
        if col == "CarbonEmission":
            continue
        if df[col].dtype == "object":
            default_row[col] = df[col].mode()[0]
        else:
            default_row[col] = df[col].median()

    user_inputs = {
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Vehicle Monthly Distance Km": vehicle_km,
        "Frequency of Traveling by Air": air,
        "Heating Energy Source": heating,
        "Diet": diet,
        "How Many New Clothes Monthly": clothes,
        "Monthly Grocery Bill": grocery,
        "Waste Bag Weekly Count": waste_count,
        "Waste Bag Size": waste_size,
        "Cooking_With": cooking_with,
        "Sex": sex,
        "Body Type": body_type,
        "Social Activity": social,
        "How Often Shower": shower,
        "How Long TV PC Daily Hour": tv_hours,
        "How Long Internet Daily Hour": internet_hours,
        "Recycling": recycling,
        "Energy efficiency": efficiency,
    }

    for key, value in user_inputs.items():
        if value not in ["Not sure", "Prefer not to say"]:
            default_row[key] = value

    user_df = pd.DataFrame([default_row])

    pred = float(model.predict(user_df)[0])

    # Results     
    st.markdown("## 📊 Your Estimated Carbon Footprint")

    percentile = (df["CarbonEmission"] < pred).mean() * 100
    score = max(0, 100 - percentile)

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Estimated carbon emission", f"{pred:.2f}")

    with colB:
        st.metric("Compared with dataset", f"Higher than {percentile:.0f}%")

    with colC:
        st.metric("Sustainability Score", f"{score:.0f}/100")

    st.caption(f"Average emission in dataset: {df['CarbonEmission'].mean():.2f}")

    if score >= 80:
        st.success("🌟 Excellent! Your lifestyle appears to have a relatively low carbon impact.")

    elif score >= 60:
        st.info(
            "👍 Your carbon footprint is moderate. "
            "A few small lifestyle adjustments could reduce it further."
        )

    elif score >= 40:
        st.warning(
            "⚠️ Your estimated carbon footprint is higher than average. "
            "Some lifestyle changes could significantly reduce your emissions."
        )

    else:
        st.error(
            "🚨 Your estimated carbon footprint is quite high compared to most users. "
            "Reducing travel emissions and improving household efficiency could make a large difference."
        )

    # Recommendation 
    st.markdown("---")
    st.markdown("## 🌱 Ways to Reduce Your Carbon Footprint")

    tips = []

    if vehicle_km > df["Vehicle Monthly Distance Km"].quantile(0.75):

        if transport == "private":
            tips.append(
                "🚗 Your driving distance is higher than most users. "
                "Reducing car trips or carpooling can significantly lower emissions."
            )

        else:
            tips.append(
                "🚗 Your travel distance is relatively high. "
                "Combining trips or using low-carbon transport options could help."
            )

    if air in ["frequently", "very frequently"]:
        tips.append(
            "✈️ Air travel contributes heavily to carbon emissions. "
            "Reducing flights when possible can make a large impact."
        )

    if clothes > df["How Many New Clothes Monthly"].quantile(0.75):
        tips.append(
            "👕 Buying clothing more frequently than average increases environmental impact. "
            "Choosing durable clothes and buying fewer items can reduce emissions."
        )

    if grocery > df["Monthly Grocery Bill"].quantile(0.5)and diet == "omnivore":
        tips.append(
            "🛒 Higher consumption levels can increase environmental impact. "
            "Reducing food waste and choosing sustainable products may help."
        )

    if waste_count > df["Waste Bag Weekly Count"].quantile(0.75):
        tips.append(
            "🗑 Your household waste is higher than average. "
            "Reducing single-use products and improving waste separation can help."
        )

    if recycling == "Not sure":
        tips.append(
            "♻️ Improving recycling habits can reduce landfill waste and environmental impact."
        )

    if efficiency == "No":
        tips.append(
            "💡 Improving home energy efficiency (LED lighting, efficient appliances) "
            "can significantly reduce household emissions."
        )

    if tv_hours > df["How Long TV PC Daily Hour"].quantile(0.75):
        tips.append(
            "💻 High screen time increases electricity consumption. "
            "Turning devices off when not in use can help reduce energy use."
        )

    if not tips:
        if score >= 80:
            st.success(
                "🌱 Great job! Your lifestyle already appears environmentally friendly."
            )
        elif score >= 60:
            st.info(
                "🌍 Your footprint is moderate. Small improvements in transport, energy use, "
                "or consumption could help reduce it further."
            )

        else:
            st.warning(
                "🌱 Your footprint is higher than average. Reviewing transportation habits, "
                "energy efficiency, and waste reduction strategies could help."
            )
    else:
        st.info("Here are some areas where small changes could help")
        for tip in tips:
            st.markdown(f"- {tip}")    