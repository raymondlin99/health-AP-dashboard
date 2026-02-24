
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
    page_title="Asst. Prof Health/Policy Jobs — US",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {background:#f0f4ff;border-radius:10px;padding:14px 18px;text-align:center;}
  .metric-num  {font-size:2rem;font-weight:700;color:#1a3c8f;}
  .metric-lbl  {font-size:.8rem;color:#555;margin-top:2px;}
  .section-hdr {font-size:1.15rem;font-weight:600;margin-top:1.2rem;margin-bottom:.4rem;color:#1a3c8f;}
</style>
""", unsafe_allow_html=True)

# ── R-tier colour map ────────────────────────────────────────────────────────
RTIER_COLOR = {
    "R1": [220,  38,  38],      # red
    "R2": [ 37,  99, 235],      # blue
    "R3": [ 22, 163,  74],      # green
    "(unknown)": [156,163,175], # grey
}

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing data file: {DATA_PATH}. Run the notebook first.")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    df["r_tier_f"] = df["r_tier"].fillna("(unknown)")
    df["salary_mid"] = df[["salary_min","salary_max"]].mean(axis=1)
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    # clamp to US bounds
    mask = df["lat"].between(18, 72) & df["lon"].between(-180, -60)
    df.loc[~mask, ["lat","lon"]] = np.nan
    if "posted_ago" not in df.columns:
        df["posted_ago"] = None
    if "published_date" not in df.columns:
        df["published_date"] = None
    return df

df = load_data()

# ── Sidebar filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Graduation_hat.svg/240px-Graduation_hat.svg.png", width=60)
    st.title("Filters")

    sources = sorted(df["source"].dropna().unique().tolist())
    source_sel = st.multiselect("Source", sources, default=sources)

    rtier_opts = ["R1","R2","R3","(unknown)"]
    rtier_sel  = st.multiselect("Research Tier (Carnegie)", rtier_opts, default=rtier_opts)

    states = sorted(df["state"].dropna().unique().tolist())
    state_sel = st.multiselect("State", states, default=states)

    salary_on = st.checkbox("Only jobs with salary info", value=False)

    st.markdown("---")
    st.info("💡 Use the **Search job titles** bar in the Job Listings tab to filter by keyword.")
    pulled = df["pulled_at_utc"].iloc[0][:10] if "pulled_at_utc" in df.columns else "unknown"
    st.caption(f"Data pulled: {pulled} · {len(df)} total listings")

# ── Apply filters ────────────────────────────────────────────────────────────
f = df.copy()
if source_sel:
    f = f[f["source"].isin(source_sel)]
f = f[f["r_tier_f"].isin(rtier_sel)]
if state_sel:
    f = f[(f["state"].isin(state_sel)) | (f["state"].isna())]
if salary_on:
    f = f[f["salary_min"].notna() | f["salary_max"].notna()]

# ── KPI row ──────────────────────────────────────────────────────────────────
st.markdown("## 🎓 Assistant Professor Jobs — Health / Public Health / Policy / Medicine (US)")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-num">{len(f)}</div><div class="metric-lbl">Total Openings</div></div>', unsafe_allow_html=True)
with k2:
    r1_n = (f["r_tier_f"] == "R1").sum()
    st.markdown(f'<div class="metric-card"><div class="metric-num">{r1_n}</div><div class="metric-lbl">R1 Schools</div></div>', unsafe_allow_html=True)
with k3:
    sal_known = f["salary_mid"].dropna()
    sal_str = f"${sal_known.median()/1000:.0f}K" if len(sal_known) > 0 else "N/A"
    st.markdown(f'<div class="metric-card"><div class="metric-num">{sal_str}</div><div class="metric-lbl">Median Salary</div></div>', unsafe_allow_html=True)
with k4:
    n_states = f["state"].dropna().nunique()
    st.markdown(f'<div class="metric-card"><div class="metric-num">{n_states}</div><div class="metric-lbl">States</div></div>', unsafe_allow_html=True)
with k5:
    mappable = f.dropna(subset=["lat","lon"]).shape[0]
    st.markdown(f'<div class="metric-card"><div class="metric-num">{mappable}</div><div class="metric-lbl">Mapped Locations</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Tab layout ───────────────────────────────────────────────────────────────
tab_map, tab_jobs, tab_analytics = st.tabs(["🗺️  Map", "📋  Job Listings", "📊  Market Analytics"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ════════════════════════════════════════════════════════════════════════════
with tab_map:
    m = f.dropna(subset=["lat","lon"]).copy()

    if len(m) == 0:
        st.info("No mappable locations in current filter. Try broadening the filters.")
    else:
        m["color"] = m["r_tier_f"].map(RTIER_COLOR).apply(lambda c: c if isinstance(c, list) else [156,163,175])

        def make_tooltip(r):
            sal = ""
            try:
                lo_v = r.get("salary_min")
                hi_v = r.get("salary_max")
                stype = r.get("salary_type")
                if pd.notna(lo_v) and lo_v > 0:
                    if stype == "hourly":
                        hi_s = f" – ${hi_v:,.2f}" if pd.notna(hi_v) and hi_v > lo_v else ""
                        sal = f"<br>⚠️ ${lo_v:,.2f}{hi_s}/hr (hourly rate)"
                    else:
                        hi_s = f" – ${hi_v:,.0f}" if pd.notna(hi_v) and hi_v != lo_v else ""
                        sal = f"<br>💰 ${lo_v:,.0f}{hi_s}"
            except Exception:
                pass
            inst  = r.get("institution") or ""
            loc   = r.get("location_raw") or ""
            tier  = r.get("r_tier_f") or ""
            title = r.get("title") or ""
            ago   = r.get("posted_ago") or ""
            ago_str = f"<br>🕐 {ago}" if ago else ""
            return f"<b>{title[:80]}</b><br>{inst}<br>📍 {loc}<br>🏫 {tier}{sal}{ago_str}"

        m["tooltip_html"] = m.apply(make_tooltip, axis=1)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=m[["lat","lon","color","tooltip_html","title","institution","location_raw","r_tier_f"]],
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=40000,
            radius_min_pixels=6,
            radius_max_pixels=22,
            pickable=True,
            auto_highlight=True,
            opacity=0.85,
        )

        view = pdk.ViewState(latitude=38.5, longitude=-96.0, zoom=3.4, pitch=0)

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"html": "{tooltip_html}", "style": {"background":"white","color":"#222","padding":"8px","border-radius":"6px","font-size":"13px","max-width":"320px"}},
        )
        st.pydeck_chart(deck, use_container_width=True, height=520)

        leg_cols = st.columns(len(RTIER_COLOR))
        for col, (tier, rgb) in zip(leg_cols, RTIER_COLOR.items()):
            hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
            col.markdown(f'<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{hex_c};margin-right:5px;vertical-align:middle;"></span>{tier}', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — JOB LISTINGS
# ════════════════════════════════════════════════════════════════════════════
with tab_jobs:
    # ── Inline title search bar ──────────────────────────────────────────────
    col_search, col_clear = st.columns([5, 1])
    with col_search:
        title_search = st.text_input(
            "🔍 Search job titles",
            placeholder='e.g. "policy", "epidemiology", "tenure-track"...',
            label_visibility="collapsed",
        )
    with col_clear:
        if st.button("Clear", use_container_width=True):
            title_search = ""

    f2 = f.copy()
    if title_search.strip():
        needle = title_search.strip().lower()
        f2 = f2[f2["title"].fillna("").str.lower().str.contains(re.escape(needle), na=False)]
        st.caption(f"**{len(f2)}** results matching **\"{title_search}\"** — clear search to see all")

    def fmt_salary(row):
        lo   = row.get("salary_min")
        hi   = row.get("salary_max")
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

    f2["Salary"] = f2.apply(fmt_salary, axis=1)

    show_cols = ["title", "institution", "location_raw", "state", "r_tier_f",
                 "Salary", "posted_ago", "published_date", "source", "link"]
    show_cols = [c for c in show_cols if c in f2.columns]

    rename_map = {
        "title": "Job Title", "institution": "Institution",
        "location_raw": "Location", "state": "State",
        "r_tier_f": "R-Tier", "posted_ago": "Posted",
        "published_date": "Date Posted", "source": "Source",
        "link": "Apply",
    }
    display_df = f2[show_cols].rename(columns=rename_map)
    display_df = display_df.sort_values(by=["Date Posted"], ascending=False, na_position="last")

    st.markdown(f'<div class="section-hdr">Showing {len(display_df)} positions</div>', unsafe_allow_html=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=540,
        column_config={
            "Apply": st.column_config.LinkColumn("Apply", display_text="🔗 Apply"),
            "Salary": st.column_config.TextColumn("Salary", width="medium"),
            "Posted": st.column_config.TextColumn("Posted", width="small"),
            "R-Tier": st.column_config.TextColumn("R-Tier", width="small"),
        },
    )

    csv = display_df.drop(columns=["Apply"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "health_jobs.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    color_map = {k: "#{:02x}{:02x}{:02x}".format(*v) for k,v in RTIER_COLOR.items()}
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
        fig_tier.update_traces(textposition="outside")
        fig_tier.update_layout(showlegend=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig_tier, use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">Jobs by State (top 15)</div>', unsafe_allow_html=True)
        state_counts = f["state"].dropna().value_counts().head(15).reset_index()
        state_counts.columns = ["State","Count"]
        fig_state = px.bar(
            state_counts, x="Count", y="State", orientation="h",
            color="Count", color_continuous_scale="Blues",
            template="plotly_white", text="Count",
        )
        fig_state.update_traces(textposition="outside")
        fig_state.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig_state, use_container_width=True)

    # ── Salary distribution ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Salary Distribution (postings that disclosed salary)</div>', unsafe_allow_html=True)
    sal_df = f[f["salary_mid"].notna() & (f["salary_mid"] > 10000)].copy()

    if len(sal_df) < 2:
        st.info("Not enough salary data in current filter to plot. Most postings do not disclose salary.")
    else:
        fig_sal = px.histogram(
            sal_df, x="salary_mid", nbins=20,
            color="r_tier_f", color_discrete_map=color_map,
            barmode="overlay", opacity=0.75,
            labels={"salary_mid":"Midpoint Salary ($)","r_tier_f":"R-Tier"},
            template="plotly_white",
        )
        fig_sal.update_layout(margin=dict(t=20,b=20), legend_title="R-Tier")
        fig_sal.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig_sal, use_container_width=True)

        sal_stats = sal_df.groupby("r_tier_f")["salary_mid"].agg(["count","median","min","max"]).reset_index()
        sal_stats.columns = ["R-Tier","# with salary","Median","Min","Max"]
        for col in ["Median","Min","Max"]:
            sal_stats[col] = sal_stats[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
        st.dataframe(sal_stats, use_container_width=True, hide_index=True)

    # ── US Choropleth ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Geographic Concentration (jobs per state)</div>', unsafe_allow_html=True)
    state_all = f["state"].dropna().value_counts().reset_index()
    state_all.columns = ["State","Count"]
    fig_choro = px.choropleth(
        state_all, locations="State", locationmode="USA-states",
        color="Count", scope="usa",
        color_continuous_scale="YlOrRd",
        labels={"Count":"# Openings"},
        template="plotly_white",
    )
    fig_choro.update_layout(margin=dict(t=10,b=10), geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_choro, use_container_width=True)

    # ── Top institutions ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Top Hiring Institutions</div>', unsafe_allow_html=True)
    inst_counts = f["institution"].dropna().value_counts().head(12).reset_index()
    inst_counts.columns = ["Institution","Openings"]
    fig_inst = px.bar(
        inst_counts, x="Openings", y="Institution", orientation="h",
        color="Openings", color_continuous_scale="Teal",
        template="plotly_white", text="Openings",
    )
    fig_inst.update_traces(textposition="outside")
    fig_inst.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(t=20,b=20))
    st.plotly_chart(fig_inst, use_container_width=True)

st.markdown("---")
st.caption("Sources: HigherEdJobs RSS · Academic Jobs Online RSS · Carnegie Classification (2021 basic codes) · Geocoding via OpenStreetMap Nominatim · Salary best-effort from posting text.")
