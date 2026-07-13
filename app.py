import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CartRescue AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- THEME & TIMES NEW ROMAN FONTS -----------------
BG = "#0E1013"
SURFACE = "#181B21"
SURFACE_ALT = "#1F232B"
BORDER = "#2A2E37"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#B7BAC1"
ACCENT = "#00D09C"
ACCENT_DIM = "#0B8F6C"
NEGATIVE = "#EB5B3C"
WARNING = "#F5A623"
INFO_BLUE = "#4C8DFF"
FONT = "'Times New Roman', Times, serif"

st.markdown(f"""
<style>
    /* Force Times New Roman globally */
    html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stButton, .stDataFrame,
    input, textarea, select, label, .stTabs, .stMetric, .stAlert, p, span, div, h1, h2, h3, h4, h5, h6 {{
        font-family: {FONT} !important;
    }}
    .stApp {{ background-color: {BG}; color: {TEXT_PRIMARY}; }}
    header[data-testid="stHeader"] {{ background-color: {BG}; }}
    
    /* Make all default text white and clearly visible */
    p, span, label, .stMarkdown, h1, h2, h3, h4, h5, h6 {{ color: {TEXT_PRIMARY} !important; }}
    
    .main-title {{
        font-weight: 700;
        color: {ACCENT} !important;
        text-align: center;
        margin-bottom: 5px;
    }}
    .subtitle {{
        text-align: center;
        font-size: 16px;
        color: {TEXT_SECONDARY} !important;
        margin-bottom: 30px;
    }}
    .metric-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    .metric-card h4 {{
        color: {TEXT_SECONDARY} !important;
        margin: 0;
    }}
    div[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    div[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stSidebar"] .stSlider label, div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stTextInput label {{
        color: {TEXT_SECONDARY} !important;
        font-size: 14px;
    }}
    
    /* Clean white coloring for DataFrames and Metrics */
    [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY} !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY} !important; }}
    .stDataFrame {{ color: {TEXT_PRIMARY} !important; }}
    
    /* Custom Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {SURFACE}; border-radius: 8px 8px 0 0; color: {TEXT_SECONDARY} !important; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}
    .stTabs [data-baseweb="tab"] p {{ color: inherit !important; }}

    /* Bright green button theme */
    .stButton > button {{ background-color: {ACCENT}; color: #06251C !important; font-weight: 700; border-radius: 8px; border: none; }}
    .stButton > button:hover {{ background-color: {ACCENT_DIM}; color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)

# ----------------- INDIAN E-COMMERCE DATA GENERATOR -----------------
@st.cache_data
def generate_ecommerce_data(n_samples=1000):
    np.random.seed(42)
    session_duration = np.random.gamma(shape=3, scale=2, size=n_samples) * 30
    cart_value = np.random.exponential(scale=3500, size=n_samples) + 499
    num_items = np.random.randint(1, 8, size=n_samples)
    scroll_depth = np.random.uniform(15, 100, size=n_samples)
    device_type = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])     # 75% Mobile in India
    is_returning_user = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    abandon_prob = (
        0.35 * (cart_value > 8000) + 0.25 * (session_duration < 45) +
        0.20 * (scroll_depth < 40) + 0.15 * (device_type == 1) -
        0.10 * (is_returning_user == 1) + np.random.normal(0, 0.1, n_samples)
    )
    abandon_prob = np.clip((abandon_prob - abandon_prob.min()) / (abandon_prob.max() - abandon_prob.min()), 0, 1)
    abandoned = (abandon_prob > 0.55).astype(int)

    dates = pd.to_datetime(datetime.now().date()) - pd.to_timedelta(np.random.randint(0, 30, size=n_samples), unit="D")
    skus = np.random.choice(
        ["Classic Cotton Kurta", "Running Shoes Pro", "Wireless Earbuds X2", "Stainless Steel Bottle",
         "Denim Jacket", "Smart Fitness Band", "Ceramic Cookware Set", "Leather Wallet"], size=n_samples
    )

    df = pd.DataFrame({
        'Date': dates, 'SKU': skus, 'Session_Duration_Sec': session_duration.astype(int),
        'Cart_Value_INR': np.round(cart_value, 2), 'Num_Items': num_items, 'Scroll_Depth_Pct': np.round(scroll_depth, 1),
        'Device_Mobile': device_type, 'Is_Returning': is_returning_user, 'Abandoned': abandoned
    })
    return df

df_raw = generate_ecommerce_data()

# ----------------- MACHINE LEARNING PIPELINE -----------------
@st.cache_resource
def train_prediction_model(data):
    X = data[['Session_Duration_Sec', 'Cart_Value_INR', 'Num_Items', 'Scroll_Depth_Pct', 'Device_Mobile', 'Is_Returning']]
    y = data['Abandoned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy, model.feature_importances_

model, model_accuracy, feature_importances = train_prediction_model(df_raw)

# ----------------- HEADER SECTION -----------------
st.write('<h1 class="main-title">🛒 CartRescue AI Engine</h1>', unsafe_allow_html=True)
st.write('<p class="subtitle">Smart Dashboard jo check karega ki customer kab bina khareede dukan se bhaag rahe hain</p>', unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🕹️ Live Session Simulator")
st.sidebar.write("Yahan se data badlein:")

input_cart_val = st.sidebar.slider("Cart me kitne ka saaman hai (₹)", min_value=500, max_value=25000, value=4500, step=500)
input_items = st.sidebar.slider("Kitne items add kiye hain?", min_value=1, max_value=10, value=3)
input_duration = st.sidebar.slider("Customer kitne seconds se website par hai?", min_value=10, max_value=600, value=120, step=10)
input_scroll = st.sidebar.slider("Usne kitna percent page scroll kiya?", min_value=0, max_value=100, value=60)
input_device = st.sidebar.selectbox("Customer kis device se shopping kar raha hai?", ["Mobile", "Desktop / Laptop"])
input_returning = st.sidebar.selectbox("Kya ye customer pehle bhi aaya hai?", ["New shopper", "Returning shopper"])
commission_rate = st.sidebar.slider("Marketplace commission (%)", min_value=5, max_value=25, value=15, step=1)

device_val = 1 if input_device == "Mobile" else 0
returning_val = 1 if input_returning == "Returning shopper" else 0

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live Customer Check", "📊 Store Insights & Data", "💰 Profit & Recovery Calculator"])

# ----------------- TAB 1: LIVE CUSTOMER CHECK -----------------
with tab1:
    st.subheader("Ek Live Customer ka Behavior Check Karein")
    
    live_features = pd.DataFrame({
        'Session_Duration_Sec': [input_duration], 'Cart_Value_INR': [input_cart_val],
        'Num_Items': [input_items], 'Scroll_Depth_Pct': [input_scroll],
        'Device_Mobile': [device_val], 'Is_Returning': [returning_val]
    })
    prob_checkout = model.predict_proba(live_features)[0][0]
    prob_abandon = model.predict_proba(live_features)[0][1]

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_color = NEGATIVE if prob_abandon > 0.7 else WARNING if prob_abandon > 0.4 else ACCENT
        st.markdown(f"""<div class="metric-card"><h4>Bhaagne ka Chance (Risk)</h4>
            <h1 style='color:{risk_color}; margin:10px 0 0 0; font-size:36px;'>{prob_abandon:.1%}</h1></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h4>Khareedne ka Chance</h4>
            <h1 style='color:{ACCENT}; margin:10px 0 0 0; font-size:36px;'>{prob_checkout:.1%}</h1></div>""", unsafe_allow_html=True)
    with col3:
        status_label = "🔴 HIGH RISK" if prob_abandon > 0.7 else "🟡 WATCHING CLOSELY" if prob_abandon > 0.4 else "🟢 SECURE BUYER"
        status_color = NEGATIVE if prob_abandon > 0.7 else WARNING if prob_abandon > 0.4 else ACCENT
        st.markdown(f"""<div class="metric-card"><h4>Alert Status</h4>
            <h2 style='color:{status_color}; margin:15px 0 0 0;'>{status_label}</h2></div>""", unsafe_allow_html=True)

    net_margin_factor = 1 - (commission_rate / 100)
    revenue_at_risk = input_cart_val * prob_abandon * net_margin_factor
    st.write("")
    st.write(f"Revenue at risk (net of {commission_rate}% commission): **₹{revenue_at_risk:,.0f}**")

    st.write("---")
    st.subheader("💡 Website Action Plan (Customer ko rokne ke liye kya karein?)")
    if prob_abandon >= 0.7:
        st.error(f"Offer a ₹{int(input_cart_val * 0.10)} coupon and surface a live chat assistant before this cart is lost.")
    elif 0.4 <= prob_abandon < 0.7:
        st.warning("Highlight free shipping incentives and display secure payment badges near checkout.")
    else:
        st.success("No intervention needed — preserve product margin and avoid extra discounts.")

# ----------------- TAB 2: STORE INSIGHTS & DATA -----------------
with tab2:
    st.subheader("Hamari Online Dukan ka Data Summary")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Visitors Checked", len(df_raw))
    with m_col2:
        st.metric("Overall Abandonment Rate", f"{df_raw['Abandoned'].mean()*100:.1f}%")
    with m_col3:
        total_lost_inr = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum() * net_margin_factor
        st.metric("Fasa Hua Paisa (Lost Revenue)", f"₹{total_lost_inr:,.0f}")
    with m_col4:
        st.metric("AI Model Accuracy", f"{model_accuracy*100:.1f}%")

    st.write("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        feat_df = pd.DataFrame({
            'Reason': ['Session Duration', 'Cart Value (₹)', 'Item Count', 'Scroll Depth', 'Mobile Device', 'Returning Shopper'],
            'Impact Score': feature_importances
        }).sort_values('Impact Score', ascending=True)
        fig_feat = px.bar(feat_df, x='Impact Score', y='Reason', orientation='h', title='Customer ke bhaagne ki sabse badi wajah?',
                           color='Impact Score', color_continuous_scale=[ACCENT_DIM, ACCENT], template=plotly_template)
        fig_feat.update_layout(showlegend=False, height=340, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY, font_family=FONT)
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_chart2:
        df_raw['Cart_Range'] = pd.cut(df_raw['Cart_Value_INR'], bins=[0, 2000, 5000, 10000, 50000],
                                       labels=['Under ₹2K', '₹2K–5K', '₹5K–10K', '₹10K+'])
        cart_grouped = df_raw.groupby('Cart_Range', observed=True)['Abandoned'].mean().reset_index()
        cart_grouped['Abandonment (%)'] = cart_grouped['Abandoned'] * 100
        fig_cart = px.bar(cart_grouped, x='Cart_Range', y='Abandonment (%)', title='Cart ki Keemat ke hisab se Dropouts',
                           color_discrete_sequence=[ACCENT], template=plotly_template)
        fig_cart.update_layout(height=340, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY, font_family=FONT)
        st.plotly_chart(fig_cart, use_container_width=True)

# ----------------- TAB 3: PROFIT & RECOVERY CALCULATOR -----------------
with tab3:
    st.subheader("Is AI App se Dukan ka kitna fayda hoga? (Simulator)")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        est_recovery_rate = st.slider("Hum kitne % bhaagne wale customers ko wapas rok lenge?", min_value=5, max_value=50, value=20, step=5)
        est_avg_discount = st.slider("Un customers ko average kitne % ka discount dena padega?", min_value=0, max_value=25, value=10, step=5)
    with col_s2:
        total_abandoned_sessions = df_raw['Abandoned'].sum()
        total_leaked_rev = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum()
        
        saved_carts = int(total_abandoned_sessions * (est_recovery_rate / 100))
        recovered_gross = total_leaked_rev * (est_recovery_rate / 100)
        discount_cost = recovered_gross * (est_avg_discount / 100)
        recovered_net = (recovered_gross - discount_cost) * net_margin_factor

        st.info(f"📈 **Business Profit Report:**\n\n"
                f"- Bache hue total orders: **{saved_carts}**\n"
                f"- Total recovered sales amount: **₹{recovered_gross:,.2f}**\n"
                f"- Discount dene me gaya kharcha: **₹{discount_cost:,.2f}**\n\n"
                f"### ✨ Net Revenue Saved (Net Profit): ₹{recovered_net:,.2f}")
