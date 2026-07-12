import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="E-Commerce Cart Abandonment Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using Streamlit markdown
st.markdown("""
<style>
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    div[data-testid="stSidebar"] {
        background-color: #f1f3f5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SYNTHETIC DATA GENERATOR -----------------
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generates realistic e-commerce session data for model training"""
    np.random.seed(42)
    
    # Generate features
    session_duration = np.random.gamma(shape=3, scale=2, size=n_samples) * 30  # seconds
    cart_value = np.random.exponential(scale=120, size=n_samples) + 15         # USD
    num_items = np.random.randint(1, 10, size=n_samples)
    scroll_depth = np.random.uniform(10, 100, size=n_samples)                  # percentage
    device_type = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])       # 0: Desktop, 1: Mobile
    is_returning_user = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) # 0: New, 1: Returning
    
    # Define abandonment logic (simulating actual customer behaviors)
    # Higher cart value, low session duration, high mobile rate, and low scroll depth lead to higher abandonment
    abandon_prob = (
        0.3 * (cart_value > 150) +
        0.2 * (session_duration < 60) +
        0.2 * (scroll_depth < 40) +
        0.15 * (device_type == 1) -
        0.1 * (is_returning_user == 1) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Scale probabilities between 0 and 1
    abandon_prob = np.clip((abandon_prob - abandon_prob.min()) / (abandon_prob.max() - abandon_prob.min()), 0, 1)
    abandoned = (abandon_prob > 0.52).astype(int)
    
    df = pd.DataFrame({
        'Session_Duration_Sec': session_duration.astype(int),
        'Cart_Value_USD': np.round(cart_value, 2),
        'Num_Items': num_items,
        'Scroll_Depth_Pct': np.round(scroll_depth, 1),
        'Device_Mobile': device_type,
        'Is_Returning': is_returning_user,
        'Abandoned': abandoned
    })
    return df

# Load simulated store data
df_raw = generate_synthetic_data()

# ----------------- MACHINE LEARNING PIPELINE -----------------
@st.cache_resource
def train_prediction_model(data):
    """Trains a quick Random Forest model to predict user checkout exits"""
    X = data.drop(columns=['Abandoned'])
    y = data['Abandoned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate simple validation accuracy
    accuracy = model.score(X_test, y_test)
    return model, accuracy, model.feature_importances_, X.columns

model, model_accuracy, feature_importances, feature_names = train_prediction_model(df_raw)

# ----------------- USER INTERFACE STRUCTURE -----------------
st.write('<h1 class="main-title">🛒 E-Commerce Revenue Rescue Engine</h1>', unsafe_allow_html=True)
st.write('<p class="subtitle">Real-time Cart Abandonment Predictor & Conversion Simulator powered by Machine Learning</p>', unsafe_allow_html=True)

# Main App Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Real-time Prediction Dashboard", "📈 Business & Feature Analytics", "⚙️ Simulation Sandbox"])

# ----------------- TAB 1: REAL-TIME PREDICTOR -----------------
with tab1:
    st.header("Simulate a Live Customer Session")
    st.write("Adjust the values in the sidebar to simulate user actions on your store website in real-time.")
    
    # Sidebar input sliders representing simulated user inputs
    st.sidebar.header("🕹️ Live Session Simulator")
    st.sidebar.subheader("Customer Metrics")
    
    input_cart_val = st.sidebar.slider("Cart Value ($)", min_value=10.0, max_value=500.0, value=120.0, step=5.0)
    input_items = st.sidebar.slider("Number of Items in Cart", min_value=1, max_value=15, value=3)
    input_duration = st.sidebar.slider("Session Duration (Seconds)", min_value=10, max_value=600, value=150, step=10)
    input_scroll = st.sidebar.slider("Page Scroll Depth (%)", min_value=0, max_value=100, value=65)
    
    st.sidebar.subheader("Device & User Profile")
    input_device = st.sidebar.selectbox("Device Category", ["Desktop", "Mobile Phone"])
    input_returning = st.sidebar.selectbox("User Frequency", ["First Time Visitor", "Returning Customer"])
    
    # Map category strings to mathematical model inputs
    device_val = 1 if input_device == "Mobile Phone" else 0
    returning_val = 1 if input_returning == "Returning Customer" else 0
    
    # Format current simulated session features
    live_features = pd.DataFrame({
        'Session_Duration_Sec': [input_duration],
        'Cart_Value_USD': [input_cart_val],
        'Num_Items': [input_items],
        'Scroll_Depth_Pct': [input_scroll],
        'Device_Mobile': [device_val],
        'Is_Returning': [returning_val]
    })
    
    # Make live model prediction
    prob_checkout = model.predict_proba(live_features)[0][0]  # Class 0: Complete purchase
    prob_abandon = model.predict_proba(live_features)[0][1]   # Class 1: Abandon shopping cart
    
    # Display Prediction Metrics in 3 distinct visual columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <h3>Prediction Risk</h3>
                <h1 style='color: {"#E53935" if prob_abandon > 0.7 else "#FDD835" if prob_abandon > 0.4 else "#43A047"}; margin:0;'>
                    {prob_abandon:.1%}
                </h1>
                <p style='margin:5px 0 0 0;'>Probability of Exit</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col2:
        purchase_prob_percentage = f"{prob_checkout:.1%}"
        st.markdown(
            f"""<div class="metric-card">
                <h3>Likelihood to Buy</h3>
                <h1 style='color: #1E88E5; margin:0;'>{purchase_prob_percentage}</h1>
                <p style='margin:5px 0 0 0;'>Probability of Completion</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col3:
        status_label = "🔴 HIGH EXIT RISK" if prob_abandon > 0.7 else "🟡 WATCHING CLOSELY" if prob_abandon > 0.4 else "🟢 SECURE BUYER"
        status_color = "#E53935" if prob_abandon > 0.7 else "#FDD835" if prob_abandon > 0.4 else "#43A047"
        st.markdown(
            f"""<div class="metric-card">
                <h3>Engine Alert Level</h3>
                <h2 style='color: {status_color}; margin:15px 0 0 0;'>{status_label}</h2>
                <p style='margin:15px 0 0 0;'>Based on Live Activity Patterns</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
    # Trigger Recommended Marketing / Engineering Interventions based on risk thresholds
    st.write("---")
    st.subheader("💡 Suggested Real-time Intervention Plan")
    
    if prob_abandon >= 0.7:
        st.error("🚨 **System Alert:** User is extremely likely to abandon their cart!")
        st.info("🎯 **Automated Real-time Action Recommended:** \n\n"
                "1. **Trigger Discount Popup:** Display an immediate, time-sensitive **15% Discount Code** (`RESCUE15`) before the customer closes the tab.\n"
                "2. **Activate Live Chat Assistant:** Offer interactive human support to answer sizing or checkout technical difficulties.\n"
                "3. **Show Urgency Indicators:** Highlight that cart items are in 'high demand' to prompt rapid checkouts.")
    elif 0.4 <= prob_abandon < 0.7:
        st.warning("⚠️ **System Alert:** User displays borderline behaviors. Moderate chance of checkout drop-out.")
        st.info("🎯 **Automated Real-time Action Recommended:** \n\n"
                "1. **Trigger Free Shipping Incentive:** Offer automated free shipping if they complete checkout in the next 10 minutes.\n"
                "2. **Render Trust Seals:** Dynamically present secure payment logo graphics near checkout forms.")
    else:
        st.success("✅ **System Alert:** User exhibits healthy checkout intent behaviors.")
        st.info("🎯 **Automated Real-time Action Recommended:** \n\n"
                "1. **Standard Upsell Push:** Do not offer discounts (saving company profit margin!). Instead, suggest minor related products.")

# ----------------- TAB 2: BUSINESS & FEATURES ANALYTICS -----------------
with tab2:
    st.header("Analytics & ML Model Insights")
    st.write("Understand the core drivers of abandonment on our simulated retail platform.")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("What causes users to run away?")
        # Build clean Feature Importance chart to show professors the data science aspect
        feat_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ') for f in feature_names],
            'Importance': feature_importances
        }).sort_values('Importance', ascending=True)
        
        fig_feat = px.bar(
            feat_df, 
            x='Importance', 
            y='Feature', 
            orientation='h', 
            title='Machine Learning Feature Importance Rank',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_feat.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_chart2:
        st.subheader("Abandonment by Cart Size")
        # Bin cart sizes to show trend
        df_raw['Cart_Range'] = pd.cut(df_raw['Cart_Value_USD'], bins=[0, 50, 100, 200, 500], labels=['$0-50', '$50-100', '$100-200', '$200+'])
        cart_grouped = df_raw.groupby('Cart_Range', observed=True)['Abandoned'].mean().reset_index()
        cart_grouped['Abandonment_Rate'] = cart_grouped['Abandoned'] * 100
        
        fig_cart = px.bar(
            cart_grouped, 
            x='Cart_Range', 
            y='Abandonment_Rate',
            title='Abandonment Rate (%) across Cart Value Brackets',
            labels={'Abandonment_Rate': 'Abandonment Rate (%)', 'Cart_Range': 'Cart Price Segment'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_cart.update_layout(height=350)
        st.plotly_chart(fig_cart, use_container_width=True)

    # Key Stat Cards
    st.write("---")
    st.subheader("Simulated Database High-Level Performance Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.metric("Total Sessions Analyzed", len(df_raw))
    with m_col2:
        base_abandon_rate = f"{df_raw['Abandoned'].mean():.1%}"
        st.metric("Overall Abandonment Rate", base_abandon_rate)
    with m_col3:
        total_lost = f"${df_raw[df_raw['Abandoned'] == 1]['Cart_Value_USD'].sum():,.2f}"
        st.metric("Total Revenue Leaked", total_lost)
    with m_col4:
        st.metric("Real-time Prediction Model Accuracy", f"{model_accuracy:.1%}")

# ----------------- TAB 3: SIMULATION SANDBOX -----------------
with tab3:
    st.header("Strategic Planning & Conversion Simulator")
    st.write("Adjust global shop parameters to simulate business recovery actions and calculate saved cash margins.")
    
    st.markdown("""
    When you deploy real-time triggers, you can recover a subset of abandoned carts. 
    Use this sandbox tool to simulate the ROI (Return on Investment) of deploying this predictor app.
    """)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Global Simulator Configuration")
        est_recovery_rate = st.slider("Target Recovery Rate for Triggered Sessions (%)", min_value=1, max_value=50, value=20)
        est_avg_discount = st.slider("Average Discount Offered to High-Risk Users (%)", min_value=0, max_value=30, value=10)
    
    with col_s2:
        st.subheader("Calculated Simulated Returns")
        
        total_abandoned_sessions = df_raw['Abandoned'].sum()
        total_leaked_rev = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_USD'].sum()
        
        # Calculate saved numbers
        saved_carts = int(total_abandoned_sessions * (est_recovery_rate / 100))
        recovered_gross = total_leaked_rev * (est_recovery_rate / 100)
        discount_losses = recovered_gross * (est_avg_discount / 100)
        recovered_net = recovered_gross - discount_losses
        
        st.info(f"📈 **Simulated ROI Summary:**\n\n"
                f"* **Carts Stopped from Leaving:** {saved_carts} transactions successfully rescued\n"
                f"* **Gross Saved Revenue:** `${recovered_gross:,.2f}`\n"
                f"* **Margin Loss due to discount codes:** `${discount_losses:,.2f}`\n"
                f"### ✨ Net Revenue Recovered: `${recovered_net:,.2f}`")

st.sidebar.write("---")
st.sidebar.info("🎓 **Internship Project Submission**  \n"
                "**Title:** E-Commerce Cart Abandonment Predictor  \n"
                "**Tech Stack:** Python, Streamlit, Scikit-Learn, Plotly")
```eof

---

