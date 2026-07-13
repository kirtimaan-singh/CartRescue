import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
plotly_template = "plotly_dark"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CartGuard Analytics | Seller Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- THEME (Groww-inspired dark UI) -----------------
BG = "#0E1013"
SURFACE = "#181B21"
SURFACE_ALT = "#1F232B"
BORDER = "#2A2E37"
TEXT_PRIMARY = "#F5F6F7"
TEXT_SECONDARY = "#8B8F98"
ACCENT = "#00D09C"       # Groww teal
ACCENT_DIM = "#0B8F6C"
NEGATIVE = "#EB5B3C"
WARNING = "#F5A623"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
    }}
    header[data-testid="stHeader"] {{
        background-color: {BG};
    }}
    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 22px;
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        margin-bottom: 22px;
    }}
    .topbar-left {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .logo-dot {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, {ACCENT} 0%, #2E7DE1 100%);
        display: inline-block;
    }}
    .brand-name {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 20px;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.3px;
    }}
    .brand-tag {{
        font-size: 12px;
        color: {TEXT_SECONDARY};
    }}
    .subtitle {{
        font-size: 15px;
        color: {TEXT_SECONDARY};
        margin-bottom: 24px;
    }}
    .section-heading {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 20px;
        color: {TEXT_PRIMARY};
        margin: 6px 0 14px 0;
    }}
    .metric-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        padding: 20px;
        border-radius: 14px;
        text-align: center;
    }}
    .status-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }}
    /* Sidebar */
    div[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    div[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stTextInput label,
    div[data-testid="stSidebar"] .stPassword label {{
        color: {TEXT_SECONDARY} !important;
        font-size: 13px;
    }}
    .sidebar-section-label {{
        font-size: 11px;
        letter-spacing: 1.5px;
        color: {ACCENT} !important;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 6px;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 6px;
    }}
    .seller-badge {{
        background-color: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {SURFACE};
        border-radius: 8px 8px 0 0;
        color: {TEXT_SECONDARY};
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}
    /* Buttons */
    .stButton > button {{
        background-color: {ACCENT};
        color: #06251C;
        border: none;
        font-weight: 600;
        border-radius: 8px;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT_DIM};
        color: #ffffff;
    }}
    /* Auth screen */
    .auth-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 36px 40px;
        max-width: 460px;
        margin: 40px auto;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE: AUTH -----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "seller_name" not in st.session_state:
    st.session_state.seller_name = ""

# ----------------- AUTH GATE -----------------
def render_auth_screen():
    st.markdown(
        f"""<div class="topbar">
            <div class="topbar-left">
                <span class="logo-dot"></span>
                <div>
                    <div class="brand-name">CartGuard Analytics</div>
                    <div class="brand-tag">Seller Intelligence Console</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="text-align:center; color:{TEXT_SECONDARY}; margin-bottom:20px;">'
        f'Sign in to monitor cart abandonment across your marketplace listings.</p>',
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        li_business = st.text_input("Business email or store ID", key="login_id")
        st.text_input("Password", type="password", key="login_pw")
        if st.button("Sign In", key="login_btn", use_container_width=True):
            if li_business.strip():
                st.session_state.authenticated = True
                st.session_state.seller_name = li_business.split("@")[0].strip().title() or "Seller"
                st.rerun()
            else:
                st.warning("Enter your business email or store ID to continue.")

    with signup_tab:
        su_name = st.text_input("Business / store name", key="signup_name")
        su_email = st.text_input("Business email", key="signup_email")
        st.selectbox("Primary marketplace", ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"], key="signup_mp")
        st.text_input("Create password", type="password", key="signup_pw")
        if st.button("Create Account", key="signup_btn", use_container_width=True):
            if su_name.strip() and su_email.strip():
                st.session_state.authenticated = True
                st.session_state.seller_name = su_name.strip()
                st.rerun()
            else:
                st.warning("Business name and email are required to create an account.")

    st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.authenticated:
    render_auth_screen()
    st.stop()

# ----------------- SYNTHETIC MARKETPLACE DATA -----------------
@st.cache_data
def generate_ecommerce_data(n_samples=1000):
    np.random.seed(42)

    session_duration = np.random.gamma(shape=3, scale=2, size=n_samples) * 30  # seconds
    cart_value = np.random.exponential(scale=3500, size=n_samples) + 499       # INR
    num_items = np.random.randint(1, 8, size=n_samples)
    scroll_depth = np.random.uniform(15, 100, size=n_samples)                  # percentage
    device_type = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])     # 75% mobile share
    is_returning_user = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

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

df_raw = generate_ecommerce_data()

# ----------------- MODEL PIPELINE -----------------
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

# ----------------- TOP BAR -----------------
st.markdown(
    f"""<div class="topbar">
        <div class="topbar-left">
            <span class="logo-dot"></span>
            <div>
                <div class="brand-name">CartGuard Analytics</div>
                <div class="brand-tag">Seller Intelligence Console</div>
            </div>
        </div>
        <div style="color:{TEXT_SECONDARY}; font-size:13px;">Signed in as <span style="color:{TEXT_PRIMARY}; font-weight:600;">{st.session_state.seller_name}</span></div>
    </div>""",
    unsafe_allow_html=True
)

st.markdown(
    f'<p class="subtitle">Cart abandonment intelligence for sellers on Amazon, Flipkart, Myntra and other marketplaces — '
    f'know when a shopper is about to walk away from your listing, and what it is costing you.</p>',
    unsafe_allow_html=True
)

# ----------------- SIDEBAR: SELLER DASHBOARD -----------------
st.sidebar.markdown('<p class="sidebar-section-label">SELLER PROFILE</p>', unsafe_allow_html=True)

seller_name = st.sidebar.text_input("Business / Store name", value=st.session_state.seller_name)
marketplace = st.sidebar.selectbox("Marketplace", ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"])
category = st.sidebar.selectbox(
    "Primary category",
    ["Fashion & Apparel", "Electronics", "Home & Kitchen", "Beauty & Personal Care", "Grocery"]
)
plan_tier = st.sidebar.selectbox("Seller plan tier", ["Standard", "Premium", "Enterprise"])
commission_rate = st.sidebar.slider("Marketplace commission (%)", min_value=5, max_value=25, value=15, step=1)

st.sidebar.markdown(
    f"""<div class="seller-badge">
        <div style="font-size:13px; color:{TEXT_SECONDARY};">Monitoring for</div>
        <div style="font-size:16px; font-weight:600; color:{TEXT_PRIMARY};">{seller_name}</div>
        <div style="font-size:12px; color:{ACCENT};">{marketplace} · {category} · {plan_tier}</div>
    </div>""",
    unsafe_allow_html=True
)

st.sidebar.markdown('<p class="sidebar-section-label">SESSION PARAMETERS</p>', unsafe_allow_html=True)
st.sidebar.caption("Adjust these to simulate a live shopper session on your listing.")

input_cart_val = st.sidebar.slider("Cart value (₹)", min_value=500, max_value=25000, value=4500, step=500)
input_items = st.sidebar.slider("Items in cart", min_value=1, max_value=10, value=3)
input_duration = st.sidebar.slider("Time on page (seconds)", min_value=10, max_value=600, value=120, step=10)
input_scroll = st.sidebar.slider("Page scroll depth (%)", min_value=0, max_value=100, value=60)
input_device = st.sidebar.selectbox("Device", ["Mobile", "Desktop / Laptop"])
input_returning = st.sidebar.selectbox("Shopper type", ["New shopper", "Returning shopper"])

device_val = 1 if input_device == "Mobile" else 0
returning_val = 1 if input_returning == "Returning shopper" else 0

st.sidebar.markdown("---")
if st.sidebar.button("Sign Out", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.seller_name = ""
    st.rerun()

# ----------------- MAIN NAVIGATION -----------------
tab1, tab2, tab3 = st.tabs(["Live Session Monitor", "Store Performance", "Recovery ROI Calculator"])

# ----------------- TAB 1: LIVE SESSION MONITOR -----------------
with tab1:
    st.markdown(f'<p class="section-heading">Live shopper session — {seller_name}</p>', unsafe_allow_html=True)
    st.write("Track a live shopper's behaviour on your listing and see, in real time, whether they are likely to abandon the cart.")

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

    col1, col2, col3 = st.columns(3)

    with col1:
        risk_color = NEGATIVE if prob_abandon > 0.7 else WARNING if prob_abandon > 0.4 else ACCENT
        st.markdown(
            f"""<div class="metric-card">
                <h4 style='color: {TEXT_SECONDARY}; margin:0; font-weight:500;'>Abandonment Risk</h4>
                <h1 style='color: {risk_color}; margin:10px 0 0 0; font-size: 36px;'>{prob_abandon:.1%}</h1>
            </div>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <h4 style='color: {TEXT_SECONDARY}; margin:0; font-weight:500;'>Checkout Likelihood</h4>
                <h1 style='color: {ACCENT}; margin:10px 0 0 0; font-size: 36px;'>{prob_checkout:.1%}</h1>
            </div>""",
            unsafe_allow_html=True
        )

    with col3:
        if prob_abandon > 0.7:
            status_label, status_color = "High Risk — likely to abandon", NEGATIVE
        elif prob_abandon > 0.4:
            status_label, status_color = "Moderate Risk — undecided", WARNING
        else:
            status_label, status_color = "Low Risk — likely to convert", ACCENT
        st.markdown(
            f"""<div class="metric-card">
                <h4 style='color: {TEXT_SECONDARY}; margin:0; font-weight:500;'>Session Status</h4>
                <div style="margin-top:18px;">
                    <span class="status-dot" style="background-color:{status_color};"></span>
                    <span style='color: {status_color}; font-size:16px; font-weight:600;'>{status_label}</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    # Revenue at risk for this specific seller, net of marketplace commission
    net_margin_factor = 1 - (commission_rate / 100)
    revenue_at_risk = input_cart_val * prob_abandon * net_margin_factor

    st.write("")
    st.caption(
        f"Estimated revenue at risk for this session (net of {commission_rate}% {marketplace} commission): "
        f"**₹{revenue_at_risk:,.0f}**"
    )

    st.write("---")
    st.markdown('<p class="section-heading">Recommended Seller Action</p>', unsafe_allow_html=True)

    if prob_abandon >= 0.7:
        st.error("This shopper is likely to exit without completing checkout.")
        st.info(
            "**Suggested response:**\n\n"
            f"1. Trigger an on-page offer — a ₹{int(input_cart_val * 0.10)} coupon has historically improved conversion in this segment.\n"
            "2. Surface a support/chat option in case of a payment or sizing concern.\n"
            "3. Add a scarcity cue (e.g. low stock indicator) if inventory genuinely supports it."
        )
    elif 0.4 <= prob_abandon < 0.7:
        st.warning("This shopper is undecided and may need a small nudge.")
        st.info(
            "**Suggested response:**\n\n"
            "1. Highlight free or discounted shipping if your listing qualifies.\n"
            "2. Reinforce trust signals — secure payment badge, return policy — near the checkout button."
        )
    else:
        st.success("This shopper shows strong purchase intent.")
        st.info(
            "**Suggested response:**\n\n"
            "1. No discount intervention needed — preserve margin.\n"
            "2. Consider a single relevant cross-sell recommendation at checkout."
        )

# ----------------- TAB 2: STORE PERFORMANCE -----------------
with tab2:
    st.markdown(f'<p class="section-heading">Store performance summary — {seller_name}</p>', unsafe_allow_html=True)
    st.write(f"Aggregated insight across the last 1,000 shopper sessions on your {marketplace} listings.")

    net_margin_factor = 1 - (commission_rate / 100)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Sessions Tracked", len(df_raw))
    with m_col2:
        st.metric("Abandonment Rate", f"{df_raw['Abandoned'].mean()*100:.1f}%")
    with m_col3:
        total_lost_inr = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum() * net_margin_factor
        st.metric("Net Revenue at Risk", f"₹{total_lost_inr:,.0f}")
    with m_col4:
        st.metric("Model Accuracy", f"{model_accuracy*100:.1f}%")

    st.write("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        feat_df = pd.DataFrame({
            'Factor': ['Session Duration', 'Cart Value (₹)', 'Item Count', 'Scroll Depth', 'Mobile Device', 'Returning Shopper'],
            'Impact Score': feature_importances
        }).sort_values('Impact Score', ascending=True)

        fig_feat = px.bar(
            feat_df, x='Impact Score', y='Factor', orientation='h',
            title='Leading drivers of cart abandonment',
            color='Impact Score', color_continuous_scale=[ACCENT_DIM, ACCENT],
            template=plotly_template
        )
        fig_feat.update_layout(
            showlegend=False, height=350,
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font_color=TEXT_PRIMARY
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_chart2:
        df_raw['Cart_Range'] = pd.cut(
            df_raw['Cart_Value_INR'],
            bins=[0, 2000, 5000, 10000, 50000],
            labels=['Under ₹2,000', '₹2,000–5,000', '₹5,000–10,000', '₹10,000+']
        )
        cart_grouped = df_raw.groupby('Cart_Range', observed=True)['Abandoned'].mean().reset_index()
        cart_grouped['Abandonment (%)'] = cart_grouped['Abandoned'] * 100

        fig_cart = px.bar(
            cart_grouped, x='Cart_Range', y='Abandonment (%)',
            title='Abandonment rate by cart value tier',
            color_discrete_sequence=[ACCENT],
            template=plotly_template
        )
        fig_cart.update_layout(
            height=350,
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font_color=TEXT_PRIMARY
        )
        st.plotly_chart(fig_cart, use_container_width=True)

# ----------------- TAB 3: RECOVERY ROI CALCULATOR -----------------
with tab3:
    st.markdown('<p class="section-heading">Recovery ROI Calculator</p>', unsafe_allow_html=True)
    st.write(
        f"Estimate the net revenue {seller_name} could recover by intervening on high-risk sessions "
        f"before they abandon, net of {marketplace}'s {commission_rate}% commission."
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("Simulation Inputs")
        est_recovery_rate = st.slider(
            "Share of at-risk shoppers recovered via intervention (%)", min_value=5, max_value=50, value=20, step=5
        )
        est_avg_discount = st.slider(
            "Average discount required to recover a cart (%)", min_value=0, max_value=25, value=10, step=5
        )

    with col_s2:
        st.subheader("Projected Impact")

        net_margin_factor = 1 - (commission_rate / 100)
        total_abandoned_sessions = df_raw['Abandoned'].sum()
        total_leaked_rev = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum()

        saved_carts = int(total_abandoned_sessions * (est_recovery_rate / 100))
        recovered_gross = total_leaked_rev * (est_recovery_rate / 100)
        discount_cost = recovered_gross * (est_avg_discount / 100)
        recovered_net = (recovered_gross - discount_cost) * net_margin_factor

        st.info(
            f"**Projected outcome for {seller_name}:**\n\n"
            f"- Carts recovered: **{saved_carts}** orders\n"
            f"- Gross sales recovered: **₹{recovered_gross:,.2f}**\n"
            f"- Discount cost: **₹{discount_cost:,.2f}**\n"
            f"- {marketplace} commission ({commission_rate}%) already deducted below\n\n"
            f"### Net Revenue Recovered: ₹{recovered_net:,.2f}"
        )
