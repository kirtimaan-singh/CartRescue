import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CartRescue AI - E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern styling for cleaner UI
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
        font-size: 16px;
        color: #666;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- INDIAN CONTEXT DATA GENERATOR -----------------
@st.cache_data
def generate_indian_ecommerce_data(n_samples=1000):
    np.random.seed(42)
    
    # Simulating realistic Indian shopper behaviors
    session_duration = np.random.gamma(shape=3, scale=2, size=n_samples) * 30  # seconds
    cart_value = np.random.exponential(scale=3500, size=n_samples) + 499       # INR (₹500 to ₹25,000+)
    num_items = np.random.randint(1, 8, size=n_samples)
    scroll_depth = np.random.uniform(15, 100, size=n_samples)                  # percentage
    device_type = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])     # 75% Mobile users in India
    is_returning_user = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]) # New vs Returning
    
    # Logic: High cart values, very short sessions, and deep scrolls without checkout raise risk
    abandon_prob = (
        0.35 * (cart_value > 8000) +
        0.25 * (session_duration < 45) +
        0.20 * (scroll_depth < 40) +
        0.15 * (device_type == 1) -
        0.10 * (is_returning_user == 1) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    abandon_prob = np.clip((abandon_prob - abandon_prob.min()) / (abandon_prob.max() - abandon_prob.min()), 0, 1)
    abandoned = (abandon_prob > 0.55).astype(int)
    
    df = pd.DataFrame({
        'Session_Duration_Sec': session_duration.astype(int),
        'Cart_Value_INR': np.round(cart_value, 2),
        'Num_Items': num_items,
        'Scroll_Depth_Pct': np.round(scroll_depth, 1),
        'Device_Mobile': device_type,
        'Is_Returning': is_returning_user,
        'Abandoned': abandoned
    })
    return df

df_raw = generate_indian_ecommerce_data()

# ----------------- MACHINE LEARNING PIPELINE -----------------
@st.cache_resource
def train_prediction_model(data):
    X = data.drop(columns=['Abandoned'])
    y = data['Abandoned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy, model.feature_importances_, X.columns

model, model_accuracy, feature_importances, feature_names = train_prediction_model(df_raw)

# ----------------- USER INTERFACE STRUCTURE -----------------
st.write('<h1 class="main-title">🛒 CartRescue AI</h1>', unsafe_allow_html=True)
st.write('<p class="subtitle">Smart System jo check karega ki online shopping karne wale customer kab bina khareede bhaag rahe hain</p>', unsafe_allow_html=True)

# Main Navigation Tabs (Simplified names)
tab1, tab2, tab3 = st.tabs(["🔮 Live Customer Check", "📊 Store Insights & Data", "💰 Profit & Recovery Calculator"])

# ----------------- TAB 1: LIVE CUSTOMER CHECK -----------------
with tab1:
    st.subheader("Ek Live Customer ka Behavior Check Karein")
    st.write("Left side wale panel se badlo ki customer website par kya kar raha hai, app turant result dikhayegi.")
    
    # Simple and clean Sidebar controls
    st.sidebar.header("🕹️ Customer Control Panel")
    st.sidebar.write("Yahan se metrics badlein:")
    
    input_cart_val = st.sidebar.slider("Cart me kitne ka saaman hai (₹)", min_value=500, max_value=25000, value=4500, step=500)
    input_items = st.sidebar.slider("Kitne items add kiye hain?", min_value=1, max_value=10, value=3)
    input_duration = st.sidebar.slider("Customer kitne seconds se website par hai?", min_value=10, max_value=600, value=120, step=10)
    input_scroll = st.sidebar.slider("Usne kitna percent page scroll kiya?", min_value=0, max_value=100, value=60)
    
    input_device = st.sidebar.selectbox("Customer kis device se shopping kar raha hai?", ["Mobile Phone", "Desktop/Laptop"])
    input_returning = st.sidebar.selectbox("Kya ye customer pehle bhi aaya hai?", ["Naya Customer (First Time)", "Purana Customer (Returning)"])
    
    # Mappings
    device_val = 1 if input_device == "Mobile Phone" else 0
    returning_val = 1 if input_returning == "Purana Customer (Returning)" else 0
    
    live_features = pd.DataFrame({
        'Session_Duration_Sec': [input_duration],
        'Cart_Value_INR': [input_cart_val],
        'Num_Items': [input_items],
        'Scroll_Depth_Pct': [input_scroll],
        'Device_Mobile': [device_val],
        'Is_Returning': [returning_val]
    })
    
    prob_checkout = model.predict_proba(live_features)[0][0]
    prob_abandon = model.predict_proba(live_features)[0][1]
    
    # 3 Clean Metric Display Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = "#E53935" if prob_abandon > 0.7 else "#FDD835" if prob_abandon > 0.4 else "#43A047"
        st.markdown(
            f"""<div class="metric-card">
                <h4 style='color: #555; margin:0;'>Bhaagne ka Chance (Risk)</h4>
                <h1 style='color: {risk_color}; margin:10px 0 0 0; font-size: 36px;'>{prob_abandon:.1%}</h1>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <h4 style='color: #555; margin:0;'>Khareedne ka Chance</h4>
                <h1 style='color: #1E88E5; margin:10px 0 0 0; font-size: 36px;'>{prob_checkout:.1%}</h1>
            </div>""", 
            unsafe_allow_html=True
        )
        
    with col3:
        status_label = "🔴 HIGH RISK (Bhaagne wala hai)" if prob_abandon > 0.7 else "🟡 SAFE ZONE (Normal Behavior)" if prob_abandon > 0.4 else "🟢 LOYAL BUYER (Khareed lega)"
        status_color = "#E53935" if prob_abandon > 0.7 else "#FDD835" if prob_abandon > 0.4 else "#43A047"
        st.markdown(
            f"""<div class="metric-card">
                <h4 style='color: #555; margin:0;'>Current Alert Status</h4>
                <h3 style='color: {status_color}; margin:18px 0 0 0;'>{status_label}</h3>
            </div>""", 
            unsafe_allow_html=True
        )
        
    # Action Plans in pure simple language
    st.write("---")
    st.subheader("💡 Website Action Plan (Customer ko rokne ke liye kya karein?)")
    
    if prob_abandon >= 0.7:
        st.error("🚨 **Alert:** Ye customer bina khareede tab band karne wala hai!")
        st.info("🎯 **Dukan ka Instant Action:** \n\n"
                f"1. **Pop-up Offer:** Screen par turant ek ₹{int(input_cart_val * 0.10)} ka discount coupon dikhao.\n"
                "2. **WhatsApp Support:** Ek button de do taaki agar payment fail ho rahi ho toh customer chat kar sake.\n"
                "3. **Urgency Hook:** Dikhao ki 'Ye item sirf 2 bache hain stock me' taaki wo jaldi order kare.")
    elif 0.4 <= prob_abandon < 0.7:
        st.warning("⚠️ **Warning:** Customer thoda confused lag raha hai.")
        st.info("🎯 **Dukan ka Instant Action:** \n\n"
                "1. **Free Shipping Banner:** Use dikhao ki is order par delivery fees ekdum FREE hai.\n"
                "2. **Trust Signals:** Payment page par '100% Secure Payment' aur 'Easy Returns' ka logo bada karke dikhao.")
    else:
        st.success("✅ **Sab Safe Hai:** Ye customer acche se shopping kar raha hai.")
        st.info("🎯 **Dukan ka Instant Action:** \n\n"
                "1. **Kuch mat karo:** Ise koi फालतू discount dene ki zaroori nahi hai, company ka profit margin save karo!\n"
                "2. **Cross-Sell:** Checkout page par bas ek chhota related item recommend kar do.")

# ----------------- TAB 2: STORE INSIGHTS & DATA -----------------
with tab2:
    st.header("Hamari Online Dukan ka Data Summary")
    st.write("Pichle 1,000 customers ke data se nikli hui ahem report:")
    
    # 4 Clean Indian Business Metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Visitors Checked", len(df_raw))
    with m_col2:
        st.metric("Overall Abandonment Rate", f"{df_raw['Abandoned'].mean()*100:.1f}%")
    with m_col3:
        total_lost_inr = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum()
        st.metric("Fasa Hua Paisa (Lost Revenue)", f"₹{total_lost_inr:,.0f}")
    with m_col4:
        st.metric("AI Model Accuracy", f"{model_accuracy*100:.1f}%")
        
    st.write("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Simple cleaned feature name logic
        feat_df = pd.DataFrame({
            'Reason': ['Session Duration', 'Cart Value (₹)', 'Items Count', 'Scroll Depth', 'Using Mobile', 'Is Old Customer'],
            'Impact Score': feature_importances
        }).sort_values('Impact Score', ascending=True)
        
        fig_feat = px.bar(
            feat_df, x='Impact Score', y='Reason', orientation='h',
            title='Customer ke bhaagne ki sabse badi wajah kya hai?',
            color='Impact Score', color_continuous_scale='Blues'
        )
        fig_feat.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_chart2:
        df_raw['Cart_Range'] = pd.cut(df_raw['Cart_Value_INR'], bins=[0, 2000, 5000, 10000, 50000], labels=['Choti Cart (Under ₹2k)', 'Medium (₹2k-5k)', 'Badi (₹5k-10k)', 'Premium (₹10k+)'])
        cart_grouped = df_raw.groupby('Cart_Range', observed=True)['Abandoned'].mean().reset_index()
        cart_grouped['Bhaagne Wale (%)'] = cart_grouped['Abandoned'] * 100
        
        fig_cart = px.bar(
            cart_grouped, x='Cart_Range', y='Bhaagne Wale (%)',
            title='Cart ki Keemat ke hisab se Dropouts',
            color_discrete_sequence=['#ff7f0e']
        )
        fig_cart.update_layout(height=350)
        st.plotly_chart(fig_cart, use_container_width=True)

# ----------------- TAB 3: PROFIT & RECOVERY CALCULATOR -----------------
with tab3:
    st.header("Is AI App se Dukan ka kitna fayda hoga? (Simulator)")
    st.write("Agar hum high-risk customers ko real-time me rokhein, toh kitna munafa (profit) wapas laya ja sakta hai:")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Simulate Karein")
        est_recovery_rate = st.slider("Hum discount/offers dekar kitne % bhaagne wale customers ko wapas rok lenge?", min_value=5, max_value=50, value=20, step=5)
        est_avg_discount = st.slider("Un customers ko average kitne % ka discount dena padega?", min_value=0, max_value=25, value=10, step=5)
    
    with col_s2:
        st.subheader("💰 Kitna Paisa Bachaya?")
        
        total_abandoned_sessions = df_raw['Abandoned'].sum()
        total_leaked_rev = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum()
        
        saved_carts = int(total_abandoned_sessions * (est_recovery_rate / 100))
        recovered_gross = total_leaked_rev * (est_recovery_rate / 100)
        discount_losses = recovered_gross * (est_avg_discount / 100)
        recovered_net = recovered_gross - discount_losses
        
        st.info(f"📈 **Business Profit Report:**\n\n"
                f"* **Bache hue total orders:** {saved_carts} orders chhutne se bach gaye\n"
                f"* **Total recovered sales amount:** `₹{recovered_gross:,.2f}`\n"
                f"* **Discount dene me gaya kharcha:** `₹{discount_losses:,.2f}`\n"
                f"### ✨ Net Revenue Saved (Net Profit): `₹{recovered_net:,.2f}`")

st.sidebar.write("---")
st.sidebar.info("🎓 **Final Internship Project**  \n"
                "**Title:** CartRescue AI Engine  \n"
                "**Tech Used:** Python, Streamlit, Machine Learning")
