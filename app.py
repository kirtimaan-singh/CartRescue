import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
plotly_template = "plotly_dark"
import io

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CartGuard Analytics | Seller Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- THEME (Groww-inspired dark UI, Times New Roman typography) -----------------
BG = "#0E1013"
SURFACE = "#181B21"
SURFACE_ALT = "#1F232B"
BORDER = "#2A2E37"
TEXT_PRIMARY = "#F5F6F7"
TEXT_SECONDARY = "#8B8F98"
ACCENT = "#00D09C"
ACCENT_DIM = "#0B8F6C"
NEGATIVE = "#EB5B3C"
WARNING = "#F5A623"
INFO_BLUE = "#4C8DFF"
FONT = "'Times New Roman', Times, serif"

st.markdown(f"""
<style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stButton, .stDataFrame,
    input, textarea, select, label, .stTabs, .stMetric, .stAlert {{
        font-family: {FONT} !important;
    }}
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
    .brand-name {{ font-family: {FONT}; font-weight: 700; font-size: 20px; color: {TEXT_PRIMARY}; }}
    .brand-tag {{ font-size: 12px; color: {TEXT_SECONDARY}; }}
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
    .section-heading {{ font-family: {FONT}; font-weight: 700; font-size: 21px; color: {TEXT_PRIMARY}; margin: 6px 0 4px 0; }}
    .section-sub {{ font-size: 13px; color: {TEXT_SECONDARY}; margin-bottom: 14px; }}
    .timestamp-row {{ font-size: 12.5px; color: {TEXT_SECONDARY}; margin-bottom: 16px; }}
    .timestamp-row .dot {{ color: {ACCENT}; }}

    .metric-card {{ background-color: {SURFACE}; border: 1px solid {BORDER}; padding: 20px; border-radius: 14px; text-align: center; }}
    .status-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}

    .panel {{ background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }}
    .alert-row {{ display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid {BORDER}; font-size: 14px; }}
    .alert-row:last-child {{ border-bottom: none; }}
    .alert-time {{ color: {TEXT_SECONDARY}; font-size: 12.5px; margin-left: auto; white-space: nowrap; }}
    .benchmark-bar-bg {{ background-color: {SURFACE_ALT}; border-radius: 6px; height: 8px; width: 100%; overflow: hidden; }}
    .benchmark-bar-fill {{ height: 8px; border-radius: 6px; }}

    .data-source-banner {{
        background-color: {SURFACE_ALT}; border: 1px solid {BORDER}; border-left: 3px solid {WARNING};
        border-radius: 8px; padding: 10px 16px; font-size: 13px; color: {TEXT_SECONDARY}; margin-bottom: 18px;
    }}
    .data-source-banner.live {{ border-left-color: {ACCENT}; }}

    div[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
    div[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; font-family: {FONT} !important; }}
    div[data-testid="stSidebar"] .stSlider label, div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stTextInput label, div[data-testid="stSidebar"] .stFileUploader label {{
        color: {TEXT_SECONDARY} !important; font-size: 13px;
    }}
    .sidebar-section-label {{
        font-size: 11.5px; letter-spacing: 1.5px; color: {ACCENT} !important; font-weight: 700;
        margin-top: 18px; margin-bottom: 6px; border-bottom: 1px solid {BORDER}; padding-bottom: 6px;
    }}
    .seller-badge {{ background-color: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px; margin-bottom: 10px; }}
    .usage-bar-bg {{ background-color: {BG}; border-radius: 6px; height: 7px; width: 100%; overflow: hidden; margin-top: 6px; }}
    .usage-bar-fill {{ height: 7px; border-radius: 6px; background-color: {ACCENT}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {SURFACE}; border-radius: 8px 8px 0 0; color: {TEXT_SECONDARY}; font-family: {FONT}; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}

    .stButton > button {{ background-color: {ACCENT}; color: #06251C; border: none; font-weight: 700; border-radius: 8px; font-family: {FONT}; }}
    .stButton > button:hover {{ background-color: {ACCENT_DIM}; color: #ffffff; }}
    .stDownloadButton > button {{
        background-color: {SURFACE_ALT}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER};
        font-weight: 700; border-radius: 8px; font-family: {FONT};
    }}
    .stDownloadButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

    .auth-wrapper {{
        display: flex; border: 1px solid {BORDER}; border-radius: 18px; overflow: hidden;
        max-width: 920px; margin: 30px auto; background-color: {SURFACE};
    }}
    .auth-brand-panel {{
        flex: 1; padding: 44px 36px; background: linear-gradient(160deg, #0B2A22 0%, #0E1013 75%);
        display: flex; flex-direction: column; justify-content: center; border-right: 1px solid {BORDER};
    }}
    .auth-form-panel {{ flex: 1.15; padding: 44px 40px; }}
    .auth-feature-row {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 16px; font-size: 14px; color: {TEXT_SECONDARY}; }}
    .auth-feature-row .tick {{ color: {ACCENT}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
for key, default in [
    ("authenticated", False), ("seller_name", ""), ("marketplace_default", "Amazon"),
    ("last_refresh", datetime.now()), ("uploaded_df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------- AUTH SCREEN -----------------
def render_auth_screen():
    st.markdown(
        f"""<div class="topbar"><div class="topbar-left"><span class="logo-mark">C</span>
            <div><div class="brand-name">CartGuard Analytics</div><div class="brand-tag">Seller Intelligence Console</div></div>
        </div></div>""", unsafe_allow_html=True
    )
    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown(f"""
        <div class="auth-brand-panel">
            <div class="logo-mark" style="margin-bottom:22px;">C</div>
            <div style="font-family:{FONT}; font-weight:700; font-size:24px; color:{TEXT_PRIMARY}; margin-bottom:10px;">
                Know the moment a sale is about to slip away.
            </div>
            <div style="font-size:14px; color:{TEXT_SECONDARY}; margin-bottom:26px; line-height:1.6;">
                CartGuard analyzes your seller performance reports from Amazon, Flipkart, Myntra and more —
                and flags where you're losing revenue to abandoned carts.
            </div>
            <div class="auth-feature-row"><span class="tick">✓</span> Works with your real Seller Central / Seller Hub reports</div>
            <div class="auth-feature-row"><span class="tick">✓</span> Revenue-at-risk, net of marketplace commission</div>
            <div class="auth-feature-row"><span class="tick">✓</span> SKU-level abandonment breakdown</div>
            <div class="auth-feature-row"><span class="tick">✓</span> Category benchmarking</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="auth-form-panel">', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:{FONT}; font-weight:700; font-size:19px; color:{TEXT_PRIMARY}; margin-bottom:4px;">Welcome back</div>'
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

# ----------------- REAL SELLER REPORT SCHEMA -----------------
# Mirrors Amazon's "Detail Page Sales and Traffic by Child Item" business report
# and Flipkart's Seller Hub performance export.
REQUIRED_COLUMNS = ["Date", "SKU", "Sessions", "Page_Views", "Units_Ordered", "Ordered_Product_Sales"]

@st.cache_data
def generate_sample_report(n_days=30):
    """Sample data shaped exactly like a real Business Report, for preview before upload."""
    np.random.seed(7)
    skus = ["Classic Cotton Kurta", "Running Shoes Pro", "Wireless Earbuds X2", "Stainless Steel Bottle",
            "Denim Jacket", "Smart Fitness Band", "Ceramic Cookware Set", "Leather Wallet",
            "Yoga Mat Premium", "Bluetooth Speaker Mini"]
    rows = []
    for d in range(n_days):
        date = (datetime.now() - timedelta(days=n_days - d)).date()
        for sku in skus:
            sessions = np.random.poisson(35)
            page_views = int(sessions * np.random.uniform(1.2, 1.8))
            conv_rate = np.clip(np.random.normal(0.18, 0.06), 0.03, 0.5)
            units = max(0, int(sessions * conv_rate))
            avg_price = np.random.uniform(400, 3200)
            sales = round(units * avg_price, 2)
            rows.append([date, sku, sessions, page_views, units, sales])
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

def parse_uploaded_report(file):
    """Parse an uploaded CSV and map common column-name variants to our schema."""
    raw = pd.read_csv(file)
    colmap_candidates = {
        "Date": ["date", "report date", "day"],
        "SKU": ["sku", "product", "product name", "child asin", "asin"],
        "Sessions": ["sessions", "sessions - total", "sessions total"],
        "Page_Views": ["page views", "page views - total", "pageviews"],
        "Units_Ordered": ["units ordered", "units_ordered", "orders", "units sold"],
        "Ordered_Product_Sales": ["ordered product sales", "sales", "revenue", "gmv"]
    }
    lower_cols = {c.lower().strip(): c for c in raw.columns}
    mapped = {}
    for target, variants in colmap_candidates.items():
        for v in variants:
            if v in lower_cols:
                mapped[target] = lower_cols[v]
                break
    missing = [c for c in REQUIRED_COLUMNS if c not in mapped]
    if missing:
        return None, missing
    df = raw[[mapped[c] for c in REQUIRED_COLUMNS]].copy()
    df.columns = REQUIRED_COLUMNS
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Sessions", "Page_Views", "Units_Ordered", "Ordered_Product_Sales"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Sessions", "Units_Ordered"])
    return df, []

# ----------------- SIDEBAR -----------------
st.sidebar.markdown('<p class="sidebar-section-label">SELLER PROFILE</p>', unsafe_allow_html=True)
seller_name = st.sidebar.text_input("Business / Store name", value=st.session_state.seller_name)
marketplace = st.sidebar.selectbox(
    "Marketplace", ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"],
    index=["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"].index(st.session_state.marketplace_default)
    if st.session_state.marketplace_default in ["Amazon", "Flipkart", "Myntra", "Meesho", "Ajio"] else 0
)
category = st.sidebar.selectbox("Primary category", ["Fashion & Apparel", "Electronics", "Home & Kitchen", "Beauty & Personal Care", "Grocery"])
plan_tier = st.sidebar.selectbox("Seller plan tier", ["Standard", "Premium", "Enterprise"])
commission_rate = st.sidebar.slider("Marketplace commission (%)", min_value=5, max_value=25, value=15, step=1)

st.sidebar.markdown('<p class="sidebar-section-label">YOUR REPORT DATA</p>', unsafe_allow_html=True)
st.sidebar.caption(f"Upload your {marketplace} Business Report / Seller Hub performance export (CSV).")
uploaded_file = st.sidebar.file_uploader("Upload report CSV", type=["csv"])

if uploaded_file is not None:
    parsed_df, missing_cols = parse_uploaded_report(uploaded_file)
    if parsed_df is None:
        st.sidebar.error(f"Could not find required columns: {', '.join(missing_cols)}. "
                          f"Expected something like Date, SKU, Sessions, Page Views, Units Ordered, Sales.")
        st.session_state.uploaded_df = None
    else:
        st.session_state.uploaded_df = parsed_df
        st.sidebar.success(f"Loaded {len(parsed_df):,} rows from your report.")

using_real_data = st.session_state.uploaded_df is not None
df_source = st.session_state.uploaded_df if using_real_data else generate_sample_report()

plan_limits = {"Standard": 5000, "Premium": 20000, "Enterprise": 100000}
sessions_used = min(int(df_source["Sessions"].sum()), plan_limits[plan_tier])
usage_pct = min(100, int(sessions_used / plan_limits[plan_tier] * 100))

st.sidebar.markdown(
    f"""<div class="seller-badge">
        <div style="font-size:13px; color:{TEXT_SECONDARY};">Monitoring for</div>
        <div style="font-size:16px; font-weight:700; color:{TEXT_PRIMARY};">{seller_name}</div>
        <div style="font-size:12.5px; color:{ACCENT};">{marketplace} · {category} · {plan_tier}</div>
        <div style="font-size:12px; color:{TEXT_SECONDARY}; margin-top:10px;">Monthly usage: {sessions_used:,} / {plan_limits[plan_tier]:,} sessions</div>
        <div class="usage-bar-bg"><div class="usage-bar-fill" style="width:{usage_pct}%;"></div></div>
    </div>""", unsafe_allow_html=True
)

st.sidebar.markdown("---")
sc1, sc2 = st.sidebar.columns(2)
with sc1:
    if st.button("Refresh", use_container_width=True):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
with sc2:
    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.seller_name = ""
        st.session_state.uploaded_df = None
        st.rerun()

# ----------------- DERIVED METRICS FROM REAL REPORT STRUCTURE -----------------
net_margin_factor = 1 - (commission_rate / 100)
df_source = df_source.copy()
df_source["Conversion_Rate"] = np.where(df_source["Sessions"] > 0, df_source["Units_Ordered"] / df_source["Sessions"], 0)
df_source["Non_Converting_Sessions"] = (df_source["Sessions"] - df_source["Units_Ordered"]).clip(lower=0)
df_source["Avg_Order_Value"] = np.where(df_source["Units_Ordered"] > 0, df_source["Ordered_Product_Sales"] / df_source["Units_Ordered"], 0)
overall_avg_order_value = df_source.loc[df_source["Units_Ordered"] > 0, "Avg_Order_Value"].mean() if (df_source["Units_Ordered"] > 0).any() else 0
df_source["Revenue_At_Risk"] = df_source["Non_Converting_Sessions"] * overall_avg_order_value * net_margin_factor
seller_abandon_rate = 100 * (1 - df_source["Conversion_Rate"].replace([np.inf, -np.inf], 0).mean())

# ----------------- TOP BAR -----------------
st.markdown(
    f"""<div class="topbar">
        <div class="topbar-left"><span class="logo-mark">C</span>
            <div><div class="brand-name">CartGuard Analytics</div><div class="brand-tag">Seller Intelligence Console</div></div>
        </div>
        <div class="topbar-right">
            <span class="plan-chip">Plan: <b>{plan_tier}</b></span>
            <span class="icon-btn" title="Settings">⚙</span>
            <span class="icon-btn" title="Team">◎</span>
            <div style="color:{TEXT_SECONDARY}; font-size:13px;">Signed in as <span style="color:{TEXT_PRIMARY}; font-weight:700;">{seller_name}</span></div>
        </div>
    </div>""", unsafe_allow_html=True
)
st.markdown(
    f'<p class="subtitle">Cart abandonment intelligence for sellers on Amazon, Flipkart, Myntra and other marketplaces — '
    f'know where you are losing revenue, and what to do about it.</p>', unsafe_allow_html=True
)

if using_real_data:
    st.markdown(
        f'<div class="data-source-banner live">Showing analysis of <b>your uploaded report</b> '
        f'({len(df_source):,} rows, {df_source["Date"].min().date()} to {df_source["Date"].max().date()}). '
        f'Last refreshed: {st.session_state.last_refresh.strftime("%d %b %Y, %I:%M %p")}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="data-source-banner">You are viewing <b>sample data</b> — upload your {marketplace} Business Report / '
        f'Seller Hub export from the sidebar to see analysis of your actual store.</div>',
        unsafe_allow_html=True
    )

# ----------------- MAIN NAVIGATION -----------------
tab1, tab2, tab3 = st.tabs(["Store Performance", "Product & Trends", "Recovery ROI Calculator"])

# ----------------- TAB 1: STORE PERFORMANCE -----------------
with tab1:
    st.markdown(f'<p class="section-heading">Store performance summary — {seller_name}</p>', unsafe_allow_html=True)
    st.write(f"Computed from {'your uploaded' if using_real_data else 'sample'} {marketplace} report "
             f"covering {df_source['Date'].nunique()} days and {df_source['SKU'].nunique()} products.")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Sessions", f"{int(df_source['Sessions'].sum()):,}")
    with m_col2:
        st.metric("Non-Conversion Rate", f"{seller_abandon_rate:.1f}%")
    with m_col3:
        st.metric("Net Revenue at Risk", f"₹{df_source['Revenue_At_Risk'].sum():,.0f}")
    with m_col4:
        st.metric("Total Sales Captured", f"₹{df_source['Ordered_Product_Sales'].sum():,.0f}")

    st.write("---")
    category_benchmarks = {"Fashion & Apparel": 82.0, "Electronics": 86.0, "Home & Kitchen": 84.5,
                            "Beauty & Personal Care": 80.5, "Grocery": 88.0}
    benchmark_non_conv = 100 - category_benchmarks.get(category, 84.0)
    delta = seller_abandon_rate - benchmark_non_conv
    delta_color = NEGATIVE if delta > 0 else ACCENT
    delta_label = f"{'+' if delta > 0 else ''}{delta:.1f} pts vs category"

    st.markdown('<p class="section-heading">Category Benchmark</p>', unsafe_allow_html=True)
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown(f"""<div class="panel"><div style="font-size:13px; color:{TEXT_SECONDARY}; margin-bottom:6px;">{seller_name} — {category}</div>
            <div class="benchmark-bar-bg"><div class="benchmark-bar-fill" style="width:{min(seller_abandon_rate,100)}%; background-color:{ACCENT};"></div></div>
            <div style="font-size:20px; font-weight:700; color:{TEXT_PRIMARY}; margin-top:8px;">{seller_abandon_rate:.1f}%</div></div>""", unsafe_allow_html=True)
    with bcol2:
        st.markdown(f"""<div class="panel"><div style="font-size:13px; color:{TEXT_SECONDARY}; margin-bottom:6px;">{marketplace} category average</div>
            <div class="benchmark-bar-bg"><div class="benchmark-bar-fill" style="width:{min(benchmark_non_conv,100)}%; background-color:{INFO_BLUE};"></div></div>
            <div style="font-size:20px; font-weight:700; color:{TEXT_PRIMARY}; margin-top:8px;">{benchmark_non_conv:.1f}%
                <span style="font-size:13px; color:{delta_color}; font-weight:700;"> ({delta_label})</span></div></div>""", unsafe_allow_html=True)

    st.write("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        sku_perf = df_source.groupby("SKU").agg(Sessions=("Sessions", "sum"), Units=("Units_Ordered", "sum")).reset_index()
        sku_perf["Conversion (%)"] = (sku_perf["Units"] / sku_perf["Sessions"] * 100).round(1)
        sku_perf = sku_perf.sort_values("Conversion (%)")
        fig_feat = px.bar(sku_perf, x="Conversion (%)", y="SKU", orientation="h",
                           title="Conversion rate by product", color="Conversion (%)",
                           color_continuous_scale=[NEGATIVE, WARNING, ACCENT], template=plotly_template)
        fig_feat.update_layout(showlegend=False, height=380, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY,
                                font_family=FONT)
        st.plotly_chart(fig_feat, use_container_width=True)
    with col_chart2:
        aov_bins = pd.cut(df_source["Avg_Order_Value"].replace(0, np.nan), bins=[0, 500, 1500, 3000, 100000],
                           labels=["Under ₹500", "₹500–1,500", "₹1,500–3,000", "₹3,000+"])
        df_source["_aov_bin"] = aov_bins
        bin_group = df_source.groupby("_aov_bin", observed=True).agg(
            Sessions=("Sessions", "sum"), Units=("Units_Ordered", "sum")
        ).reset_index()
        bin_group["Non-Conversion (%)"] = (100 * (1 - bin_group["Units"] / bin_group["Sessions"])).round(1)
        fig_cart = px.bar(bin_group, x="_aov_bin", y="Non-Conversion (%)", title="Non-conversion rate by order value tier",
                           color_discrete_sequence=[ACCENT], template=plotly_template)
        fig_cart.update_layout(height=380, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font_color=TEXT_PRIMARY,
                                font_family=FONT, xaxis_title="Order value tier")
        st.plotly_chart(fig_cart, use_container_width=True)

    st.write("---")
    st.markdown('<p class="section-heading">Export</p>', unsafe_allow_html=True)
    csv_data = df_source.drop(columns=["_aov_bin"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Download analyzed data (CSV)", data=csv_data,
                        file_name=f"cartguard_{seller_name.replace(' ', '_')}_analysis.csv", mime="text/csv")

# ----------------- TAB 2: PRODUCT & TRENDS -----------------
with tab2:
    st.markdown(f'<p class="section-heading">Product & trend insight — {seller_name}</p>', unsafe_allow_html=True)
    st.write("Daily trend and product-level breakdown computed directly from your report.")

    trend_df = df_source.groupby(df_source["Date"].dt.date).agg(
        Sessions=("Sessions", "sum"), Units=("Units_Ordered", "sum")
    ).reset_index().sort_values("Date")
    trend_df["Non-Conversion Rate (%)"] = (100 * (1 - trend_df["Units"] / trend_df["Sessions"])).round(1)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_df["Date"], y=trend_df["Non-Conversion Rate (%)"], mode="lines",
        line=dict(color=ACCENT, width=2.5), fill="tozeroy", fillcolor="rgba(0,208,156,0.10)",
        name="Non-Conversion Rate"
    ))
    fig_trend.update_layout(title="Non-conversion rate over time", height=330, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                             font_color=TEXT_PRIMARY, font_family=FONT, template=plotly_template, margin=dict(t=50, b=30))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.write("---")
    col_sku, col_alert = st.columns([1.3, 1])
    with col_sku:
        st.markdown('<p class="section-heading" style="font-size:16px;">Top products by revenue at risk</p>', unsafe_allow_html=True)
        sku_summary = df_source.groupby("SKU").agg(
            Sessions=("Sessions", "sum"), Units_Ordered=("Units_Ordered", "sum"),
            Revenue_At_Risk=("Revenue_At_Risk", "sum")
        ).reset_index().sort_values("Revenue_At_Risk", ascending=False).head(8)
        sku_summary.columns = ["Product", "Sessions", "Units Ordered", "Revenue at Risk (₹)"]
        st.dataframe(sku_summary.style.format({"Revenue at Risk (₹)": "₹{:,.0f}"}), use_container_width=True, hide_index=True)

    with col_alert:
        st.markdown('<p class="section-heading" style="font-size:16px;">Products needing attention</p>', unsafe_allow_html=True)
        worst = df_source.groupby("SKU").agg(Sessions=("Sessions", "sum"), Units=("Units_Ordered", "sum")).reset_index()
        worst["Conversion (%)"] = (worst["Units"] / worst["Sessions"] * 100).round(1)
        worst = worst.sort_values("Conversion (%)").head(4)
        rows = ""
        for _, r in worst.iterrows():
            rows += (f'<div class="alert-row"><span class="status-dot" style="background-color:{NEGATIVE};"></span>'
                     f'{r["SKU"]} — {r["Conversion (%)"]:.1f}% conversion'
                     f'<span class="alert-time">{int(r["Sessions"])} sessions</span></div>')
        st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

# ----------------- TAB 3: RECOVERY ROI CALCULATOR -----------------
with tab3:
    st.markdown('<p class="section-heading">Recovery ROI Calculator</p>', unsafe_allow_html=True)
    st.write(f"Estimate the net revenue {seller_name} could recover by improving conversion on low-performing "
             f"sessions, net of {marketplace}'s {commission_rate}% commission.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Simulation Inputs")
        est_recovery_rate = st.slider("Share of non-converting sessions recovered via intervention (%)", min_value=5, max_value=50, value=20, step=5)
        est_avg_discount = st.slider("Average discount required to recover a sale (%)", min_value=0, max_value=25, value=10, step=5)

    with col_s2:
        st.subheader("Projected Impact")
        total_non_converting = df_source["Non_Converting_Sessions"].sum()
        total_leaked_rev = total_non_converting * overall_avg_order_value
        saved_orders = int(total_non_converting * (est_recovery_rate / 100))
        recovered_gross = total_leaked_rev * (est_recovery_rate / 100)
        discount_cost = recovered_gross * (est_avg_discount / 100)
        recovered_net = (recovered_gross - discount_cost) * net_margin_factor

        st.info(f"**Projected outcome for {seller_name}:**\n\n"
                f"- Orders recovered: **{saved_orders:,}**\n"
                f"- Gross sales recovered: **₹{recovered_gross:,.2f}**\n"
                f"- Discount cost: **₹{discount_cost:,.2f}**\n"
                f"- {marketplace} commission ({commission_rate}%) already deducted below\n\n"
                f"### Net Revenue Recovered: ₹{recovered_net:,.2f}")
