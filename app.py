
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = os.path.join("jobs_dashboard", "jobs_enriched.parquet")

st.set_page_config(
    page_title="Health Faculty Jobs · US",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ────────────────────────────────────────────────────────────
BRAND      = "#1e3a5f"       # deep navy
ACCENT     = "#2563eb"       # vivid blue
ACCENT2    = "#0ea5e9"       # sky blue
SUCCESS    = "#059669"       # emerald
WARN       = "#d97706"       # amber
DANGER     = "#dc2626"       # red
BG_CARD    = "#f8faff"
BG_SIDEBAR = "#0f172a"       # very dark navy for sidebar

st.markdown(f"""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
  }}

  /* ── Page background ── */
  .stApp {{
    background: linear-gradient(160deg, #eef2ff 0%, #f8faff 60%, #e0f2fe 100%);
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BG_SIDEBAR} 0%, #1e293b 100%) !important;
  }}
  [data-testid="stSidebar"] * {{
    color: #e2e8f0 !important;
  }}
  [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
    background: {ACCENT} !important;
  }}
  [data-testid="stSidebar"] hr {{
    border-color: #334155 !important;
  }}
  [data-testid="stSidebar"] .stCheckbox label {{
    color: #e2e8f0 !important;
  }}

  /* ── Hero banner ── */
  .hero {{
    background: linear-gradient(135deg, {BRAND} 0%, {ACCENT} 60%, {ACCENT2} 100%);
    border-radius: 18px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(37,99,235,0.18);
  }}
  .hero h1 {{
    color: #fff;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
  }}
  .hero p {{
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin: 0;
  }}

  /* ── KPI cards ── */
  .kpi-grid {{
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }}
  .kpi-card {{
    flex: 1;
    min-width: 130px;
    background: #fff;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(30,58,95,0.08);
    border-top: 4px solid {ACCENT};
    text-align: center;
    transition: transform .15s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37,99,235,0.14); }}
  .kpi-card.green  {{ border-top-color: {SUCCESS}; }}
  .kpi-card.amber  {{ border-top-color: {WARN}; }}
  .kpi-card.sky    {{ border-top-color: {ACCENT2}; }}
  .kpi-card.red    {{ border-top-color: {DANGER}; }}
  .kpi-icon  {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .kpi-num   {{ font-size: 2rem; font-weight: 800; color: {BRAND}; line-height: 1.1; }}
  .kpi-lbl   {{ font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }}

  /* ── Section headers ── */
  .section-hdr {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {BRAND};
    margin: 1.4rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section-hdr::after {{
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT}44, transparent);
    border-radius: 2px;
  }}

  /* ── R-tier badges ── */
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: .4px;
  }}
  .badge-R1      {{ background:#fee2e2; color:#991b1b; }}
  .badge-R2      {{ background:#dbeafe; color:#1e40af; }}
  .badge-R3      {{ background:#d1fae5; color:#065f46; }}
  .badge-unknown {{ background:#f1f5f9; color:#475569; }}

  /* ── Search box ── */
  .search-wrap input {{
    border-radius: 12px !important;
    border: 2px solid {ACCENT}55 !important;
    font-size: 0.95rem !important;
    padding: 10px 16px !important;
    background: #fff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.07) !important;
    transition: border .2s;
  }}
  .search-wrap input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT}22 !important;
  }}

  /* ── Tabs ── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0;
  }}
  [data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    color: #64748b;
    padding: 10px 22px;
    border: none;
    transition: all .15s;
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
    background: {ACCENT} !important;
    color: #fff !important;
    box-shadow: 0 -2px 12px {ACCENT}44;
  }}

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(30,58,95,0.07);
    border: 1px solid #e2e8f0;
  }}

  /* ── Download button ── */
  [data-testid="stDownloadButton"] button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 20px;
    box-shadow: 0 2px 8px {ACCENT}44;
    transition: opacity .15s;
  }}
  [data-testid="stDownloadButton"] button:hover {{ opacity: .88; }}

  /* ── Map legend chips ── */
  .legend-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    margin: 3px;
  }}
  .legend-dot {{
    width: 12px; height: 12px;
    border-radius: 50%;
    display: inline-block;
  }}

  /* ── Results badge ── */
  .result-count {{
    display: inline-block;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    color: white;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 10px;
  }}
</style>
""", unsafe_allow_html=True)

# ── R-tier colour map ────────────────────────────────────────────────────────
RTIER_COLOR = {
    "R1":        [220,  38,  38],
    "R2":        [ 37,  99, 235],
    "R3":        [ 22, 163,  74],
    "(unknown)": [156, 163, 175],
}

CITY_SIZE_ORDER = [
    "Major metro (1M+)",
    "Large city (500K-1M)",
    "Mid-size city (100K-500K)",
    "Small city (50K-100K)",
    "Town (< 50K)",
    "Unknown",
]

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing data file: {DATA_PATH}. Run the notebook first.")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    df["r_tier_f"]   = df["r_tier"].fillna("(unknown)")
    df["salary_mid"] = df[["salary_min","salary_max"]].mean(axis=1)
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    mask = df["lat"].between(18, 72) & df["lon"].between(-180, -60)
    df.loc[~mask, ["lat","lon"]] = np.nan
    for col, default in [("posted_ago", None), ("published_date", None),
                         ("city_size", "Unknown")]:
        if col not in df.columns:
            df[col] = default
    if "city_pop" not in df.columns:
        df["city_pop"] = np.nan
    df["city_size"] = df["city_size"].fillna("Unknown")
    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 12px;">
      <div style="font-size:2.8rem;">🎓</div>
      <div style="font-size:1.1rem;font-weight:800;color:#fff;letter-spacing:-.3px;">Health Faculty Jobs</div>
      <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">US · Assistant Professor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#334155;margin:4px 0 16px;"></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">📡 Source</p>', unsafe_allow_html=True)
    sources    = sorted(df["source"].dropna().unique().tolist())
    source_sel = st.multiselect("Source", sources, default=sources, label_visibility="collapsed")

    st.markdown('<p style="font-size:.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin:14px 0 6px;">🏫 Research Tier</p>', unsafe_allow_html=True)
    rtier_opts = ["R1","R2","R3","(unknown)"]
    rtier_sel  = st.multiselect("Research Tier", rtier_opts, default=rtier_opts, label_visibility="collapsed")

    st.markdown('<p style="font-size:.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin:14px 0 6px;">📍 State</p>', unsafe_allow_html=True)
    states    = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.multiselect("State", states, default=states, label_visibility="collapsed")

    st.markdown('<p style="font-size:.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin:14px 0 6px;">🏙️ City Size</p>', unsafe_allow_html=True)
    city_size_sel = st.multiselect("City size", options=CITY_SIZE_ORDER, default=CITY_SIZE_ORDER, label_visibility="collapsed")

    st.markdown('<div style="height:1px;background:#334155;margin:16px 0;"></div>', unsafe_allow_html=True)

    salary_on = st.checkbox("💰  Only jobs with salary info", value=False)

    st.markdown('<div style="height:1px;background:#334155;margin:16px 0 10px;"></div>', unsafe_allow_html=True)

    pulled = df["pulled_at_utc"].iloc[0][:10] if "pulled_at_utc" in df.columns else "unknown"
    st.markdown(f'<div style="font-size:.72rem;color:#64748b;text-align:center;">Data pulled: {pulled}<br>{len(df)} total listings</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.7rem;color:#475569;text-align:center;margin-top:8px;">💡 Use the search bar<br>in Job Listings to filter by title</div>', unsafe_allow_html=True)

# ── Apply filters ────────────────────────────────────────────────────────────
f = df.copy()
if source_sel:
    f = f[f["source"].isin(source_sel)]
f = f[f["r_tier_f"].isin(rtier_sel)]
if state_sel:
    f = f[(f["state"].isin(state_sel)) | (f["state"].isna())]
if salary_on:
    f = f[f["salary_min"].notna() | f["salary_max"].notna()]
if city_size_sel and len(city_size_sel) < len(CITY_SIZE_ORDER):
    f = f[f["city_size"].isin(city_size_sel)]

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 Health &amp; Policy Faculty Jobs — US</h1>
  <p>Assistant Professor openings in Public Health · Health Policy · Epidemiology · Medicine · Biostatistics</p>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
sal_known = f["salary_mid"].dropna()
sal_str   = f"${sal_known.median()/1000:.0f}K" if len(sal_known) > 0 else "N/A"
r1_n      = int((f["r_tier_f"] == "R1").sum())
n_states  = int(f["state"].dropna().nunique())
mappable  = int(f.dropna(subset=["lat","lon"]).shape[0])

k1, k2, k3, k4, k5 = st.columns(5)
cards = [
    (k1, "📋", len(f),     "Total Openings",  ""),
    (k2, "🔴", r1_n,       "R1 Schools",       "red"),
    (k3, "💰", sal_str,    "Median Salary",    "green"),
    (k4, "🗺️", n_states,  "States",            "sky"),
    (k5, "📍", mappable,   "Mapped Locations", "amber"),
]
for col, icon, val, lbl, cls in cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-num">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_jobs, tab_analytics = st.tabs(["🗺️  Map View", "📋  Job Listings", "📊  Market Analytics"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ════════════════════════════════════════════════════════════════════════════
with tab_map:
    m = f.dropna(subset=["lat","lon"]).copy()
    if len(m) == 0:
        st.info("No mappable locations match the current filters.")
    else:
        m["color"] = m["r_tier_f"].map(RTIER_COLOR).apply(lambda c: c if isinstance(c, list) else [156,163,175])

        def make_tooltip(r):
            sal = ""
            try:
                lo_v  = r.get("salary_min")
                hi_v  = r.get("salary_max")
                stype = r.get("salary_type")
                if pd.notna(lo_v) and lo_v > 0:
                    if stype == "hourly":
                        hi_s = f" – ${hi_v:,.2f}" if pd.notna(hi_v) and hi_v > lo_v else ""
                        sal  = f"<br>⚠️ ${lo_v:,.2f}{hi_s}/hr"
                    else:
                        hi_s = f" – ${hi_v:,.0f}" if pd.notna(hi_v) and hi_v != lo_v else ""
                        sal  = f"<br>💰 ${lo_v:,.0f}{hi_s}"
            except Exception:
                pass
            pop = r.get("city_pop")
            try:
                pop_str = f"<br>🏙️ Pop. {int(pop):,}" if pd.notna(pop) and pop > 0 else ""
            except Exception:
                pop_str = ""
            ago     = r.get("posted_ago") or ""
            ago_str = f"<br>🕐 {ago}" if ago else ""
            title   = (r.get("title") or "")[:80]
            inst    = r.get("institution") or ""
            loc     = r.get("location_raw") or ""
            tier    = r.get("r_tier_f") or ""
            return f"<b>{title}</b><br>{inst}<br>📍 {loc}{pop_str}<br>🏫 {tier}{sal}{ago_str}"

        m["tooltip_html"] = m.apply(make_tooltip, axis=1)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=m[["lat","lon","color","tooltip_html"]],
            get_position=["lon","lat"],
            get_fill_color="color",
            get_radius=38000,
            radius_min_pixels=6,
            radius_max_pixels=24,
            pickable=True,
            auto_highlight=True,
            opacity=0.88,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=38.5, longitude=-96.0, zoom=3.4, pitch=0),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"html": "{tooltip_html}", "style": {
                "background":"white","color":"#1e293b",
                "padding":"10px 14px","border-radius":"10px",
                "font-size":"13px","max-width":"300px",
                "box-shadow":"0 4px 16px rgba(0,0,0,0.12)",
                "border":"1px solid #e2e8f0",
            }},
        )
        st.pydeck_chart(deck, use_container_width=True, height=520)

        # Legend
        leg_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:10px;">'
        tier_labels = {"R1":"R1 — Very High Research","R2":"R2 — High Research","R3":"R3 — Doctoral","(unknown)":"Unknown Tier"}
        for tier, rgb in RTIER_COLOR.items():
            hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
            leg_html += f'<div class="legend-chip"><span class="legend-dot" style="background:{hex_c};"></span>{tier_labels[tier]}</div>'
        leg_html += "</div>"
        st.markdown(leg_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — JOB LISTINGS
# ════════════════════════════════════════════════════════════════════════════
with tab_jobs:
    col_search, col_clear = st.columns([6, 1])
    with col_search:
        st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
        title_search = st.text_input(
            "search",
            placeholder='🔍  Search job titles — e.g. "policy", "epidemiology", "tenure-track"...',
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_clear:
        if st.button("✕ Clear", use_container_width=True):
            title_search = ""

    f2 = f.copy()
    if title_search.strip():
        needle = title_search.strip().lower()
        f2 = f2[f2["title"].fillna("").str.lower().str.contains(re.escape(needle), na=False)]
        st.markdown(f'<div class="result-count">✓ {len(f2)} results for "{title_search}"</div>', unsafe_allow_html=True)

    def fmt_salary(row):
        lo    = row.get("salary_min")
        hi    = row.get("salary_max")
        stype = row.get("salary_type")
        try:
            if pd.notna(lo) and lo > 0:
                if stype == "hourly":
                    hi_str = f" – ${hi:,.2f}" if pd.notna(hi) and hi > lo else ""
                    return f"⚠️ ${lo:,.2f}{hi_str}/hr (hourly)"
                if pd.notna(hi) and hi > lo:
                    return f"${lo:,.0f} – ${hi:,.0f}"
                return f"${lo:,.0f}"
        except Exception:
            pass
        return ""

    def fmt_pop(row):
        pop  = row.get("city_pop")
        size = row.get("city_size") or "Unknown"
        try:
            if pd.notna(pop) and pop > 0:
                return f"{int(pop):,}  ({size})"
        except Exception:
            pass
        return size if size != "Unknown" else ""

    f2["Salary"]          = f2.apply(fmt_salary, axis=1)
    f2["City Population"] = f2.apply(fmt_pop, axis=1)

    show_cols = ["title","institution","location_raw","state","r_tier_f",
                 "City Population","Salary","posted_ago","published_date","source","link"]
    show_cols = [c for c in show_cols if c in f2.columns]

    rename_map = {
        "title":"Job Title","institution":"Institution",
        "location_raw":"Location","state":"State",
        "r_tier_f":"R-Tier","posted_ago":"Posted",
        "published_date":"Date Posted","source":"Source","link":"Apply",
    }
    display_df = f2[show_cols].rename(columns=rename_map)
    display_df = display_df.sort_values("Date Posted", ascending=False, na_position="last")

    n_shown = len(display_df)
    st.markdown(f'<div class="section-hdr">Showing {n_shown} position{"s" if n_shown != 1 else ""}</div>', unsafe_allow_html=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=540,
        column_config={
            "Apply":           st.column_config.LinkColumn("Apply", display_text="🔗 Apply"),
            "City Population": st.column_config.TextColumn("City Population", width="large"),
            "Salary":          st.column_config.TextColumn("Salary", width="medium"),
            "Posted":          st.column_config.TextColumn("Posted", width="small"),
            "R-Tier":          st.column_config.TextColumn("R-Tier", width="small"),
        },
    )

    csv = display_df.drop(columns=["Apply"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download CSV", csv, "health_jobs.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    color_map = {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in RTIER_COLOR.items()}

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-hdr">Jobs by Research Tier</div>', unsafe_allow_html=True)
        tier_counts = f["r_tier_f"].value_counts().reset_index()
        tier_counts.columns = ["R-Tier","Count"]
        fig_tier = px.bar(
            tier_counts, x="R-Tier", y="Count",
            color="R-Tier", color_discrete_map=color_map,
            text="Count", template="plotly_white",
        )
        fig_tier.update_traces(textposition="outside", marker_line_width=0)
        fig_tier.update_layout(showlegend=False, margin=dict(t=20,b=20),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_tier.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_tier, use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">Jobs by State (top 15)</div>', unsafe_allow_html=True)
        state_counts = f["state"].dropna().value_counts().head(15).reset_index()
        state_counts.columns = ["State","Count"]
        fig_state = px.bar(
            state_counts, x="Count", y="State", orientation="h",
            color="Count", color_continuous_scale=["#bfdbfe","#2563eb"],
            template="plotly_white", text="Count",
        )
        fig_state.update_traces(textposition="outside", marker_line_width=0)
        fig_state.update_layout(yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False, margin=dict(t=20,b=20),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_state.update_xaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig_state, use_container_width=True)

    # ── City size distribution ───────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Jobs by City Size</div>', unsafe_allow_html=True)
    size_counts = f["city_size"].value_counts().reindex(CITY_SIZE_ORDER).dropna().reset_index()
    size_counts.columns = ["City Size","Count"]
    size_colors = ["#1e3a5f","#2563eb","#0ea5e9","#38bdf8","#bae6fd","#e2e8f0"]
    fig_size = px.bar(
        size_counts, x="Count", y="City Size", orientation="h",
        color="City Size",
        color_discrete_sequence=size_colors,
        template="plotly_white", text="Count",
    )
    fig_size.update_traces(textposition="outside", marker_line_width=0)
    fig_size.update_layout(showlegend=False, margin=dict(t=10,b=10),
                           yaxis=dict(autorange="reversed"),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_size, use_container_width=True)

    # ── Salary distribution ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Salary Distribution</div>', unsafe_allow_html=True)
    sal_df = f[f["salary_mid"].notna() & (f["salary_mid"] > 10000)].copy()
    if len(sal_df) < 2:
        st.info("Not enough salary data in current filter. Most postings do not disclose salary.")
    else:
        fig_sal = px.histogram(
            sal_df, x="salary_mid", nbins=20,
            color="r_tier_f", color_discrete_map=color_map,
            barmode="overlay", opacity=0.78,
            labels={"salary_mid":"Midpoint Salary ($)","r_tier_f":"R-Tier"},
            template="plotly_white",
        )
        fig_sal.update_layout(margin=dict(t=10,b=10), legend_title="R-Tier",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_sal.update_xaxes(tickformat="$,.0f", gridcolor="#f1f5f9")
        fig_sal.update_yaxes(gridcolor="#f1f5f9")
        fig_sal.update_traces(marker_line_width=0)
        st.plotly_chart(fig_sal, use_container_width=True)

        sal_stats = sal_df.groupby("r_tier_f")["salary_mid"].agg(["count","median","min","max"]).reset_index()
        sal_stats.columns = ["R-Tier","# with salary","Median","Min","Max"]
        for col in ["Median","Min","Max"]:
            sal_stats[col] = sal_stats[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
        st.dataframe(sal_stats, use_container_width=True, hide_index=True)

    # ── US Choropleth ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Geographic Concentration</div>', unsafe_allow_html=True)
    state_all = f["state"].dropna().value_counts().reset_index()
    state_all.columns = ["State","Count"]
    fig_choro = px.choropleth(
        state_all, locations="State", locationmode="USA-states",
        color="Count", scope="usa",
        color_continuous_scale=["#dbeafe","#1e3a5f"],
        labels={"Count":"# Openings"},
        template="plotly_white",
    )
    fig_choro.update_layout(margin=dict(t=10,b=10),
                            geo=dict(bgcolor="rgba(0,0,0,0)"),
                            paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_choro, use_container_width=True)

    # ── Top institutions ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Top Hiring Institutions</div>', unsafe_allow_html=True)
    inst_counts = f["institution"].dropna().value_counts().head(12).reset_index()
    inst_counts.columns = ["Institution","Openings"]
    fig_inst = px.bar(
        inst_counts, x="Openings", y="Institution", orientation="h",
        color="Openings", color_continuous_scale=["#bfdbfe","#1e3a5f"],
        template="plotly_white", text="Openings",
    )
    fig_inst.update_traces(textposition="outside", marker_line_width=0)
    fig_inst.update_layout(yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=False, margin=dict(t=10,b=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig_inst.update_xaxes(gridcolor="#f1f5f9")
    st.plotly_chart(fig_inst, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:32px;padding:18px 24px;background:linear-gradient(135deg,#1e3a5f,#1e293b);
     border-radius:14px;color:#94a3b8;font-size:0.75rem;text-align:center;line-height:1.8;">
  Sources: <b style="color:#cbd5e1">HigherEdJobs RSS + Keyword Search</b> ·
  <b style="color:#cbd5e1">Carnegie Classification 2021</b> · <b style="color:#cbd5e1">OpenStreetMap Nominatim</b> ·
  <b style="color:#cbd5e1">SimpleMaps US Cities</b><br>
  Salary data is best-effort extracted from posting text. Always verify on the employer's site.
</div>
""", unsafe_allow_html=True)
