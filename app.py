import streamlit as st

import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from datetime import datetime, timedelta

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

INFO_BLUE = "#4C8DFF"



st.markdown(f"""

<style>

    .stApp {{ background-color: {BG}; }}

    header[data-testid="stHeader"] {{ background-color: {BG}; }}



    .topbar {{

        display: flex; align-items: center; justify-content: space-between;

        padding: 14px 22px; background-color: {SURFACE};

        border: 1px solid {BORDER}; border-radius: 14px; margin-bottom: 18px;

    }}

    .topbar-left {{ display: flex; align-items: center; gap: 10px; }}

    .topbar-right {{ display: flex; align-items: center; gap: 18px; }}

    .logo-mark {{

        width: 38px; height: 38px; border-radius: 11px;

        background: linear-gradient(135deg, {ACCENT} 0%, #2E7DE1 100%);

        display: flex; align-items: center; justify-content: center;

        font-weight: 800; font-size: 17px; color: #06251C;

    }}

    .brand-name {{ font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 19px; color: {TEXT_PRIMARY}; letter-spacing: -0.3px; }}

    .brand-tag {{ font-size: 11.5px; color: {TEXT_SECONDARY}; }}

    .plan-chip {{

        background-color: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 20px;

        padding: 5px 14px; font-size: 12px; color: {TEXT_SECONDARY};

    }}

    .plan-chip b {{ color: {ACCENT}; }}

    .icon-btn {{

        width: 34px; height: 34px; border-radius: 9px; background-color: {SURFACE_ALT};

        border: 1px solid {BORDER}; display: flex; align-items: center; justify-content: center;

        font-size: 15px; color: {TEXT_SECONDARY};

    }}



    .subtitle {{ font-size: 15px; color: {TEXT_SECONDARY}; margin-bottom: 20px; }}

    .section-heading {{ font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 20px; color: {TEXT_PRIMARY}; margin: 6px 0 4px 0; }}

    .section-sub {{ font-size: 13px; color: {TEXT_SECONDARY}; margin-bottom: 14px; }}

    .timestamp-row {{ font-size: 12px; color: {TEXT_SECONDARY}; margin-bottom: 16px; }}

    .timestamp-row .dot {{ color: {ACCENT}; }}



    .metric-card {{ background-color: {SURFACE}; border: 1px solid {BORDER}; padding: 20px; border-radius: 14px; text-align: center; }}

    .status-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}



    .panel {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }}

    .alert-row {{

        display: flex; align-items: center; gap: 10px; padding: 10px 0;

        border-bottom: 1px solid {BORDER}; font-size: 13.5px;

    }}

    .alert-row:last-child {{ border-bottom: none; }}

    .alert-time {{ color: {TEXT_SECONDARY}; font-size: 12px; margin-left: auto; white-space: nowrap; }}

    .benchmark-bar-bg {{ background-color: {SURFACE_ALT}; border-radius: 6px; height: 8px; width: 100%; overflow: hidden; }}

    .benchmark-bar-fill {{ height: 8px; border-radius: 6px; }}



    /* Sidebar */

    div[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}

    div[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

    div[data-testid="stSidebar"] .stSlider label, div[data-testid="stSidebar"] .stSelectbox label,

    div[data-testid="stSidebar"] .stTextInput label, div[data-testid="stSidebar"] .stPassword label {{

        color: {TEXT_SECONDARY} !important; font-size: 13px;

    }}

    .sidebar-section-label {{

        font-size: 11px; letter-spacing: 1.5px; color: {ACCENT} !important; font-weight: 600;

        margin-top: 18px; margin-bottom: 6px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;

    }}

    .seller-badge {{ background-color: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px; margin-bottom: 10px; }}

    .usage-bar-bg {{ background-color: {BG}; border-radius: 6px; height: 7px; width: 100%; overflow: hidden; margin-top: 6px; }}

    .usage-bar-fill {{ height: 7px; border-radius: 6px; background-color: {ACCENT}; }}



    /* Tabs */

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}

    .stTabs [data-baseweb="tab"] {{ background-color: {SURFACE}; border-radius: 8px 8px 0 0; color: {TEXT_SECONDARY}; }}

    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}



    /* Buttons */

    .stButton > button {{ background-color: {ACCENT}; color: #06251C; border: none; font-weight: 600; border-radius: 8px; }}

    .stButton > button:hover {{ background-color: {ACCENT_DIM}; color: #ffffff; }}

    .stDownloadButton > button {{

        background-color: {SURFACE_ALT}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER};

        font-weight: 600; border-radius: 8px;

    }}

    .stDownloadButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}



    /* Auth screen */

    .auth-wrapper {{

        display: flex; border: 1px solid {BORDER}; border-radius: 18px; overflow: hidden;

        max-width: 920px; margin: 30px auto; background-color: {SURFACE};

    }}

    .auth-brand-panel {{

        flex: 1; padding: 44px 36px; background: linear-gradient(160deg, #0B2A22 0%, #0E1013 75%);

        display: flex; flex-direction: column; justify-content: center; border-right: 1px solid {BORDER};

    }}

    .auth-form-panel {{ flex: 1.15; padding: 44px 40px; }}

    .auth-feature-row {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 16px; font-size: 13.5px; color: {TEXT_SECONDARY}; }}

    .auth-feature-row .tick {{ color: {ACCENT}; font-weight: 700; }}

</style>

""", unsafe_allow_html=True)



# ----------------- SESSION STATE -----------------

for key, default in [

    ("authenticated", False),

    ("seller_name", ""),

    ("marketplace_default", "Amazon"),

    ("last_refresh", datetime.now()),

]:

    if key not in st.session_state:

        st.session_state[key] = default



# ----------------- AUTH SCREEN -----------------

def render_auth_screen():

    st.markdown(

        f"""<div class="topbar">

            <div class="topbar-left">

                <span class="logo-mark">C</span>

                <div>

                    <div class="brand-name">CartGuard Analytics</div>

                    <div class="brand-tag">Seller Intelligence Console</div>

                </div>

            </div>

        </div>""",

        unsafe_allow_html=True

    )



    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.15], gap="large")



    with left:

        st.markdown(f"""

        <div class="auth-brand-panel">

            <div class="logo-mark" style="margin-bottom:22px;">C</div>

            <div style="font-family:'Poppins',sans-serif; font-weight:700; font-size:24px; color:{TEXT_PRIMARY}; margin-bottom:10px;">

                Know the moment a sale is about to slip away.

            </div>

            <div style="font-size:13.5px; color:{TEXT_SECONDARY}; margin-bottom:26px; line-height:1.6;">

                CartGuard connects to your storefront on Amazon, Flipkart, Myntra and more —

                and flags high-risk carts before they're abandoned.

            </div>

            <div class="auth-feature-row"><span class="tick">✓</span> Real-time cart risk scoring per session</div>

            <div class="auth-feature-row"><span class="tick">✓</span> Revenue-at-risk, net of marketplace commission</div>

            <div class="auth-feature-row"><span class="tick">✓</span> SKU-level abandonment breakdown</div>

            <div class="auth-feature-row"><span class="tick">✓</span> Category benchmarking</div>

        </div>

        """, unsafe_allow_html=True)



    with right:

        st.markdown('<div class="auth-form-panel">', unsafe_allow_html=True)

        st.markdown(

            f'<div style="font-family:\'Poppins\',sans-serif; font-weight:600; font-size:19px; color:{TEXT_PRIMARY}; margin-bottom:4px;">Welcome back</div>'

            f'<div style="font-size:13px; color:{TEXT_SECONDARY}; margin-bottom:22px;">Sign in to your seller console.</div>',

            unsafe_allow_html=True

        )



        login_tab, signup_tab = st.tabs(["Sign In", "Create Account"])



        with login_tab:

            li_business = st.text_input("Business email or store ID", key="login_id", placeholder="you@yourbusiness.com")

            st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")

            c1, c2 = st.columns([1, 1])

            with c1:

                st.checkbox("Keep me signed in", key="login_remember")

            with c2:

                st.markdown(f'<div style="text-align:right; font-size:12px; color:{ACCENT}; margin-top:8px;">Forgot password?</div>', unsafe_allow_html=True)

            if st.button("Sign In", key="login_btn", use_container_width=True):

                if li_business.strip():

                    st.session_state.authenticated = True

                    st.session_state.seller_name = li_business.split("@")[0].strip().title() or "Seller"

                    st.session_state.last_refresh = datetime.now()

                    st.rerun()

                else:

                    st.warning("Enter your business email or store ID to continue.")

            st.markdown(f'<div style="text-align:center; font-size:12px; color:{TEXT_SECONDARY}; margin-top:14px;">By signing in you agree to CartGuard\'s Terms of Service and Privacy Policy.</div>', unsafe_allow_html=True)



        with signup_tab:

            su_name = st.text_input("Business / store name", key="signup_name", placeholder="Aarav Textiles Pvt. Ltd.")

            su_email = st.text_input("Business email", key="signup_email", placeholder="you@yourbusiness.com")

            su_marketplace = st.selectbox("Primary marketplace", ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"], key="signup_mp")

            st.text_input("Create password", type="password", key="signup_pw", placeholder="At least 8 characters")

            if st.button("Create Account", key="signup_btn", use_container_width=True):

                if su_name.strip() and su_email.strip():

                    st.session_state.authenticated = True

                    st.session_state.seller_name = su_name.strip()

                    st.session_state.marketplace_default = su_marketplace

                    st.session_state.last_refresh = datetime.now()

                    st.rerun()

                else:

                    st.warning("Business name and email are required to create an account.")



        st.markdown('</div>', unsafe_allow_html=True)



    st.markdown('</div>', unsafe_allow_html=True)



if not st.session_state.authenticated:

    render_auth_screen()

    st.stop()



# ----------------- SYNTHETIC MARKETPLACE DATA -----------------

@st.cache_data

def generate_ecommerce_data(n_samples=1000):

    np.random.seed(42)

    session_duration = np.random.gamma(shape=3, scale=2, size=n_samples) * 30

    cart_value = np.random.exponential(scale=3500, size=n_samples) + 499

    num_items = np.random.randint(1, 8, size=n_samples)

    scroll_depth = np.random.uniform(15, 100, size=n_samples)

    device_type = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])

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

         "Denim Jacket", "Smart Fitness Band", "Ceramic Cookware Set", "Leather Wallet", "Yoga Mat Premium",

         "Bluetooth Speaker Mini"],

        size=n_samples

    )



    df = pd.DataFrame({

        'Date': dates,

        'SKU': skus,

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

    X = data[['Session_Duration_Sec', 'Cart_Value_INR', 'Num_Items', 'Scroll_Depth_Pct', 'Device_Mobile', 'Is_Returning']]

    y = data['Abandoned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy, model.feature_importances_, X.columns



model, model_accuracy, feature_importances, feature_names = train_prediction_model(df_raw)



# ----------------- SIDEBAR -----------------

st.sidebar.markdown('<p class="sidebar-section-label">SELLER PROFILE</p>', unsafe_allow_html=True)

seller_name = st.sidebar.text_input("Business / Store name", value=st.session_state.seller_name)

store_options = [f"{seller_name} — Flagship Store", f"{seller_name} — Clearance Outlet", f"{seller_name} — New Arrivals"]

active_store = st.sidebar.selectbox("Active storefront", store_options)

marketplace = st.sidebar.selectbox(

    "Marketplace", ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"],

    index=["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"].index(st.session_state.marketplace_default)

    if st.session_state.marketplace_default in ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"] else 0

)

category = st.sidebar.selectbox("Primary category", ["Fashion & Apparel", "Electronics", "Home & Kitchen", "Beauty & Personal Care", "Grocery"])

plan_tier = st.sidebar.selectbox("Seller plan tier", ["Standard", "Premium", "Enterprise"])

commission_rate = st.sidebar.slider("Marketplace commission (%)", min_value=5, max_value=25, value=15, step=1)



plan_limits = {"Standard": 5000, "Premium": 20000, "Enterprise": 100000}

sessions_used = min(int(len(df_raw) * 1.24), plan_limits[plan_tier])

usage_pct = min(100, int(sessions_used / plan_limits[plan_tier] * 100))



st.sidebar.markdown(

    f"""<div class="seller-badge">

        <div style="font-size:13px; color:{TEXT_SECONDARY};">Monitoring for</div>

        <div style="font-size:16px; font-weight:600; color:{TEXT_PRIMARY};">{seller_name}</div>

        <div style="font-size:12px; color:{ACCENT};">{marketplace} · {category} · {plan_tier}</div>

        <div style="font-size:11.5px; color:{TEXT_SECONDARY}; margin-top:10px;">Monthly usage: {sessions_used:,} / {plan_limits[plan_tier]:,} sessions</div>

        <div class="usage-bar-bg"><div class="usage-bar-fill" style="width:{usage_pct}%;"></div></div>

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

sc1, sc2 = st.sidebar.columns(2)

with sc1:

    if st.button("Refresh Data", use_container_width=True):

        st.session_state.last_refresh = datetime.now()

        st.rerun()

with sc2:

    if st.button("Sign Out", use_container_width=True):

        st.session_state.authenticated = False

        st.session_state.seller_name = ""

        st.rerun()



# ----------------- TOP BAR -----------------

st.markdown(

    f"""<div class="topbar">

        <div class="topbar-left">

            <span class="logo-mark">C</span>

            <div>

                <div class="brand-name">CartGuard Analytics</div>

                <div class="brand-tag">Seller Intelligence Console</div>

            </div>

        </div>

        <div class="topbar-right">

            <span class="plan-chip">Plan: <b>{plan_tier}</b></span>

            <span class="plan-chip">{active_store}</span>

            <span class="icon-btn" title="Settings">⚙</span>

            <span class="icon-btn" title="Team">◎</span>

            <div style="color:{TEXT_SECONDARY}; font-size:13px;">Signed in as <span style="color:{TEXT_PRIMARY}; font-weight:600;">{seller_name}</span></div>

        </div>

    </div>""",

    unsafe_allow_html=True

)



st.markdown(

    f'<p class="subtitle">Cart abandonment intelligence for sellers on Amazon, Flipkart, Myntra and other marketplaces — '

    f'know when a shopper is about to walk away from your listing, and what it is costing you.</p>',

    unsafe_allow_html=True

)

st.markdown(

    f'<div class="timestamp-row"><span class="dot">●</span> Data last refreshed: {st.session_state.last_refresh.strftime("%d %b %Y, %I:%M %p")} · Window: last 30 days</div>',

    unsafe_allow_html=True

)



# ----------------- MAIN NAVIGATION -----------------

tab1, tab2, tab3, tab4 = st.tabs(["Live Session Monitor", "Store Performance", "Product & Trends", "Recovery ROI Calculator"])



# ----------------- TAB 1: LIVE SESSION MONITOR -----------------

with tab1:

    st.markdown(f'<p class="section-heading">Live shopper session — {active_store}</p>', unsafe_allow_html=True)

    st.write("Track a live shopper's behaviour on your listing and see, in real time, whether they are likely to abandon the cart.")



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

        st.markdown(f"""<div class="metric-card"><h4 style='color:{TEXT_SECONDARY}; margin:0; font-weight:500;'>Abandonment Risk</h4>

            <h1 style='color:{risk_color}; margin:10px 0 0 0; font-size:36px;'>{prob_abandon:.1%}</h1></div>""", unsafe_allow_html=True)

    with col2:

        st.markdown(f"""<div class="metric-card"><h4 style='color:{TEXT_SECONDARY}; margin:0; font-weight:500;'>Checkout Likelihood</h4>

            <h1 style='color:{ACCENT}; margin:10px 0 0 0; font-size:36px;'>{prob_checkout:.1%}</h1></div>""", unsafe_allow_html=True)

    with col3:

        if prob_abandon > 0.7:

            status_label, status_color = "High Risk — likely to abandon", NEGATIVE

        elif prob_abandon > 0.4:

            status_label, status_color = "Moderate Risk — undecided", WARNING

        else:

            status_label, status_color = "Low Risk — likely to convert", ACCENT

        st.markdown(f"""<div class="metric-card"><h4 style='color:{TEXT_SECONDARY}; margin:0; font-weight:500;'>Session Status</h4>

            <div style="margin-top:18px;"><span class="status-dot" style="background-color:{status_color};"></span>

            <span style='color:{status_color}; font-size:16px; font-weight:600;'>{status_label}</span></div></div>""", unsafe_allow_html=True)



    net_margin_factor = 1 - (commission_rate / 100)

    revenue_at_risk = input_cart_val * prob_abandon * net_margin_factor

    st.write("")

    st.caption(f"Estimated revenue at risk for this session (net of {commission_rate}% {marketplace} commission): **₹{revenue_at_risk:,.0f}**")



    st.write("---")

    st.markdown('<p class="section-heading">Recommended Seller Action</p>', unsafe_allow_html=True)

    if prob_abandon >= 0.7:

        st.error("This shopper is likely to exit without completing checkout.")

        st.info("**Suggested response:**\n\n"

                f"1. Trigger an on-page offer — a ₹{int(input_cart_val * 0.10)} coupon has historically improved conversion in this segment.\n"

                "2. Surface a support/chat option in case of a payment or sizing concern.\n"

                "3. Add a scarcity cue (e.g. low stock indicator) if inventory genuinely supports it.")

    elif 0.4 <= prob_abandon < 0.7:

        st.warning("This shopper is undecided and may need a small nudge.")

        st.info("**Suggested response:**\n\n"

                "1. Highlight free or discounted shipping if your listing qualifies.\n"

                "2. Reinforce trust signals — secure payment badge, return policy — near the checkout button.")

    else:

        st.success("This shopper shows strong purchase intent.")

        st.info("**Suggested response:**\n\n"

                "1. No discount intervention needed — preserve margin.\n"

                "2. Consider a single relevant cross-sell recommendation at checkout.")



    st.write("---")

    st.markdown('<p class="section-heading">Recent Alerts</p>', unsafe_allow_html=True)

    recent_high_risk = df_raw[df_raw['Abandoned'] == 1].sort_values('Date', ascending=False).head(4)

    alert_rows = ""

    now_ref = datetime.now()

    for i, (_, r) in enumerate(recent_high_risk.iterrows()):

        hrs_ago = (i + 1) * 2

        alert_rows += (

            f'<div class="alert-row"><span class="status-dot" style="background-color:{NEGATIVE};"></span>'

            f'{r["SKU"]} — ₹{r["Cart_Value_INR"]:,.0f} cart flagged as high risk'

            f'<span class="alert-time">{hrs_ago}h ago</span></div>'

        )

    st.markdown(f'<div class="panel">{alert_rows}</div>', unsafe_allow_html=True)



# ----------------- TAB 2: STORE PERFORMANCE -----------------

with tab2:

    st.markdown(f'<p class="section-heading">Store performance summary — {active_store}</p>', unsafe_allow_html=True)

    st.write(f"Aggregated insight across the last 1,000 shopper sessions on your {marketplace} listings.")



    net_margin_factor = 1 - (commission_rate / 100)

    seller_abandon_rate = df_raw['Abandoned'].mean() * 100



    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:

        st.metric("Sessions Tracked", len(df_raw))

    with m_col2:

        st.metric("Abandonment Rate", f"{seller_abandon_rate:.1f}%")

    with m_col3:

        total_lost_inr = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum() * net_margin_factor

        st.metric("Net Revenue at Risk", f"₹{total_lost_inr:,.0f}")

    with m_col4:

        st.metric("Model Accuracy", f"{model_accuracy*100:.1f}%")



    st.write("---")



    # ---- Category benchmark ----

    category_benchmarks = {

        "Fashion & Apparel": 31.5, "Electronics": 27.8, "Home & Kitchen": 29.4,

        "Beauty & Personal Care": 33.2, "Grocery": 24.6

    }

    benchmark_rate = category_benchmarks.get(category, 29.0)

    delta = seller_abandon_rate - benchmark_rate

    delta_color = NEGATIVE if delta > 0 else ACCENT

    delta_label = f"{'+' if delta > 0 else ''}{delta:.1f} pts vs category" 



    st.markdown('<p class="section-heading">Category Benchmark</p>', unsafe_allow_html=True)

    bcol1, bcol2 = st.columns(2)

    with bcol1:

        st.markdown(

            f"""<div class="panel">

                <div style="font-size:13px; color:{TEXT_SECONDARY}; margin-bottom:6px;">{seller_name} — {category}</div>

                <div class="benchmark-bar-bg"><div class="benchmark-bar-fill" style="width:{min(seller_abandon_rate,100)}%; background-color:{ACCENT};"></div></div>

                <div style="font-size:20px; font-weight:700; color:{TEXT_PRIMARY}; margin-top:8px;">{seller_abandon_rate:.1f}%</div>

            </div>""", unsafe_allow_html=True)

    with bcol2:

        st.markdown(

            f"""<div class="panel">

                <div style="font-size:13px; color:{TEXT_SECONDARY}; margin-bottom:6px;">{marketplace} category average</div>

                <div class="benchmark-bar-bg"><div class="benchmark-bar-fill" style="width:{min(benchmark_rate,100)}%; background-color:{INFO_BLUE};"></div></div>

                <div style="font-size:20px; font-weight:700; color:{TEXT_PRIMARY}; margin-top:8px;">{benchmark_rate:.1f}%

                    <span style="font-size:13px; color:{delta_color}; font-weight:600;"> ({delta_label})</span>

                </div>

            </div>""", unsafe_allow_html=True)



    st.write("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:

        feat_df = pd.DataFrame({

            'Factor': ['Session Duration', 'Cart Value (₹)', 'Item Count', 'Scroll Depth', 'Mobile Device', 'Returning Shopper'],

            'Impact Score': feature_importances

        }).sort_values('Impact Score', ascending=True)

        fig_feat = px.bar(feat_df, x='Impact Score', y='Factor', orientation='h',

                           title='Leading drivers of cart abandonment',

                           color='Impact Score', color_continuous_scale=[ACCENT_DIM, ACCENT], template=plotly_template)

        fig_feat.update_layout(showlegend=False, height=350, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY)

        st.plotly_chart(fig_feat, use_container_width=True)



    with col_chart2:

        df_raw['Cart_Range'] = pd.cut(df_raw['Cart_Value_INR'], bins=[0, 2000, 5000, 10000, 50000],

                                       labels=['Under ₹2,000', '₹2,000–5,000', '₹5,000–10,000', '₹10,000+'])

        cart_grouped = df_raw.groupby('Cart_Range', observed=True)['Abandoned'].mean().reset_index()

        cart_grouped['Abandonment (%)'] = cart_grouped['Abandoned'] * 100

        fig_cart = px.bar(cart_grouped, x='Cart_Range', y='Abandonment (%)',

                           title='Abandonment rate by cart value tier',

                           color_discrete_sequence=[ACCENT], template=plotly_template)

        fig_cart.update_layout(height=350, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY)

        st.plotly_chart(fig_cart, use_container_width=True)



    st.write("---")

    st.markdown('<p class="section-heading">Export</p>', unsafe_allow_html=True)

    csv_data = df_raw.drop(columns=['Cart_Range'], errors='ignore').to_csv(index=False).encode('utf-8')

    st.download_button("Download session data (CSV)", data=csv_data, file_name=f"cartguard_{seller_name.replace(' ', '_')}_sessions.csv", mime="text/csv")



# ----------------- TAB 3: PRODUCT & TRENDS -----------------

with tab3:

    st.markdown(f'<p class="section-heading">Product & trend insight — {active_store}</p>', unsafe_allow_html=True)

    st.write("30-day trend, product-level breakdown, and the checkout funnel for your listings.")



    # ---- Trend line ----

    trend_df = df_raw.groupby(df_raw['Date'].dt.date).agg(

        Sessions=('Abandoned', 'count'), Abandoned=('Abandoned', 'sum')

    ).reset_index()

    trend_df['Abandonment Rate (%)'] = (trend_df['Abandoned'] / trend_df['Sessions'] * 100).round(1)

    trend_df = trend_df.sort_values('Date')



    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(

        x=trend_df['Date'], y=trend_df['Abandonment Rate (%)'],

        mode='lines', line=dict(color=ACCENT, width=2.5),

        fill='tozeroy', fillcolor='rgba(0,208,156,0.10)', name='Abandonment Rate'

    ))

    fig_trend.update_layout(

        title="Abandonment rate — last 30 days", height=320,

        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY,

        template=plotly_template, margin=dict(t=50, b=30)

    )

    st.plotly_chart(fig_trend, use_container_width=True)



    st.write("---")

    col_sku, col_funnel = st.columns([1.3, 1])



    with col_sku:

        st.markdown('<p class="section-heading" style="font-size:16px;">Top products by revenue at risk</p>', unsafe_allow_html=True)

        sku_summary = df_raw[df_raw['Abandoned'] == 1].groupby('SKU').agg(

            Abandoned_Sessions=('Abandoned', 'count'), Revenue_At_Risk=('Cart_Value_INR', 'sum')

        ).reset_index()

        sku_summary['Revenue_At_Risk'] = (sku_summary['Revenue_At_Risk'] * net_margin_factor).round(0)

        sku_summary = sku_summary.sort_values('Revenue_At_Risk', ascending=False).head(8)

        sku_summary.columns = ['Product', 'Abandoned Sessions', 'Revenue at Risk (₹)']

        st.dataframe(

            sku_summary.style.format({'Revenue at Risk (₹)': '₹{:,.0f}'}),

            use_container_width=True, hide_index=True

        )



    with col_funnel:

        st.markdown('<p class="section-heading" style="font-size:16px;">Checkout funnel</p>', unsafe_allow_html=True)

        total_sessions = len(df_raw)

        funnel_stages = ["Product Viewed", "Added to Cart", "Checkout Started", "Payment Attempted", "Order Placed"]

        funnel_values = [

            int(total_sessions * 2.4), total_sessions,

            int(total_sessions * (1 - df_raw['Abandoned'].mean() * 0.55)),

            int(total_sessions * (1 - df_raw['Abandoned'].mean() * 0.80)),

            int(total_sessions * (1 - df_raw['Abandoned'].mean()))

        ]

        fig_funnel = go.Figure(go.Funnel(

            y=funnel_stages, x=funnel_values,

            marker=dict(color=[ACCENT_DIM, ACCENT, WARNING, INFO_BLUE, ACCENT]),

            textinfo="value+percent initial"

        ))

        fig_funnel.update_layout(height=350, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY, margin=dict(t=10, b=10))

        st.plotly_chart(fig_funnel, use_container_width=True)



# ----------------- TAB 4: RECOVERY ROI CALCULATOR -----------------

with tab4:

    st.markdown('<p class="section-heading">Recovery ROI Calculator</p>', unsafe_allow_html=True)

    st.write(f"Estimate the net revenue {seller_name} could recover by intervening on high-risk sessions "

             f"before they abandon, net of {marketplace}'s {commission_rate}% commission.")



    col_s1, col_s2 = st.columns(2)

    with col_s1:

        st.subheader("Simulation Inputs")

        est_recovery_rate = st.slider("Share of at-risk shoppers recovered via intervention (%)", min_value=5, max_value=50, value=20, step=5)

        est_avg_discount = st.slider("Average discount required to recover a cart (%)", min_value=0, max_value=25, value=10, step=5)



    with col_s2:

        st.subheader("Projected Impact")

        net_margin_factor = 1 - (commission_rate / 100)

        total_abandoned_sessions = df_raw['Abandoned'].sum()

        total_leaked_rev = df_raw[df_raw['Abandoned'] == 1]['Cart_Value_INR'].sum()

        saved_carts = int(total_abandoned_sessions * (est_recovery_rate / 100))

        recovered_gross = total_leaked_rev * (est_recovery_rate / 100)

        discount_cost = recovered_gross * (est_avg_discount / 100)

        recovered_net = (recovered_gross - discount_cost) * net_margin_factor



        st.info(f"**Projected outcome for {seller_name}:**\n\n"

                f"- Carts recovered: **{saved_carts}** orders\n"

                f"- Gross sales recovered: **₹{recovered_gross:,.2f}**\n"

                f"- Discount cost: **₹{discount_cost:,.2f}**\n"

                f"- {marketplace} commission ({commission_rate}%) already deducted below\n\n"

                f"### Net Revenue Recovered: ₹{recovered_net:,.2f}")


