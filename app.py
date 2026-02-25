
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
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
  /* Dark Orange theme for metric cards */
  .metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease-in-out;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(249, 115, 22, 0.15), 0 4px 6px -2px rgba(249, 115, 22, 0.1);
    border-color: #f97316;
  }
  /* Typography for numbers and labels */
  .metric-num {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f97316;
    letter-spacing: -0.025em;
    line-height: 1.2;
  }
  .metric-lbl {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 8px;
  }
  /* Dark mode section headers */
  .section-hdr {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #334155;
  }
  /* Streamlit native overrides for dark look */
  [data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ── R-tier colour map ────────────────────────────────────────────────────────
RTIER_COLOR = {
    "R1": [220,  38,  38],   # red
    "R2": [ 37, 99, 235],   # blue
    "R3": [ 22, 163,  74],  # green
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
    # ensure optional columns exist
    for col, default in [("posted_ago", None), ("published_date", None),
                         ("city_pop", np.nan), ("city_size", "Unknown"),
                         ("salary_type", None)]:
        if col not in df.columns:
            df[col] = default
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

    CITY_SIZE_ORDER = ["Major metro (1M+)", "Large city (500K-1M)",
                       "Mid-size city (100K-500K)", "Small city (50K-100K)",
                       "Town (< 50K)", "Unknown"]
    city_size_opts = [s for s in CITY_SIZE_ORDER if s in df["city_size"].unique()]
    city_size_sel = st.multiselect("🏙️ City Size", city_size_opts, default=city_size_opts)

    st.markdown("---")
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
if city_size_sel:
    f = f[f["city_size"].isin(city_size_sel)]

# ── KPI row ──────────────────────────────────────────────────────────────────
st.markdown("## 🎓 Assistant Professor + Postdoc Jobs — Policy / Health Services Research Focus (US)")

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
        RTIER_HEX = {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in RTIER_COLOR.items()}

        fmap = folium.Map(location=[38.5, -96.0], zoom_start=4,
                          tiles="CartoDB positron", control_scale=True)
        cluster = MarkerCluster(name="Jobs", options={"maxClusterRadius": 35})

        for _, r in m.iterrows():
            title = (r.get("title") or "")[:90]
            inst  = r.get("institution") or ""
            loc   = r.get("location_raw") or ""
            tier  = r.get("r_tier_f") or ""
            link  = r.get("link") or ""
            sal   = ""
            if pd.notna(r.get("salary_min")) and r["salary_min"] > 0:
                lo = f"${r['salary_min']:,.0f}"
                hi = f" – ${r['salary_max']:,.0f}" if pd.notna(r.get("salary_max")) and r["salary_max"] != r["salary_min"] else ""
                sal = f"<br>💰 {lo}{hi}"
            color = RTIER_HEX.get(tier, "#9ca3af")

            popup_html = (
                f'<div style="font-family:sans-serif;min-width:220px;max-width:320px;font-size:13px;">'
                f'<b>{title}</b><br>'
                f'{inst}<br>'
                f'📍 {loc} &nbsp; 🏫 {tier}{sal}<br><br>'
                f'<a href="{link}" target="_blank" rel="noopener" '
                f'style="background:{color};color:white;padding:6px 14px;border-radius:5px;'
                f'text-decoration:none;font-weight:600;">🔗 Open Job Page</a>'
                f'</div>'
            )

            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{title[:50]} — {inst[:30]}",
            ).add_to(cluster)

        cluster.add_to(fmap)
        st_folium(fmap, use_container_width=True, height=540)

        # Legend
        leg_cols = st.columns(len(RTIER_COLOR))
        for col, (tier, rgb) in zip(leg_cols, RTIER_COLOR.items()):
            hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
            col.markdown(f'<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{hex_c};margin-right:5px;vertical-align:middle;"></span>{tier}', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — JOB LISTINGS
# ════════════════════════════════════════════════════════════════════════════
with tab_jobs:
    # ── Prominent search bar ─────────────────────────────────────────────────
    search_col1, search_col2 = st.columns([5, 1])
    with search_col1:
        title_search = st.text_input("🔍 Search job titles", value="", placeholder="e.g. policy, epidemiology, biostatistics...")
    with search_col2:
        st.write("")  # spacer
        if st.button("✕ Clear", use_container_width=True):
            title_search = ""

    f2 = f.copy()
    if title_search.strip():
        needle = title_search.strip().lower()
        f2 = f2[f2["title"].fillna("").str.lower().str.contains(re.escape(needle), na=False)]
        st.info(f"🔎 Showing **{len(f2)}** jobs matching **\"{title_search.strip()}\"**")

    # Build salary display string
    def fmt_salary(row):
        lo = row.get("salary_min")
        hi = row.get("salary_max")
        txt = row.get("salary_text")
        stype = row.get("salary_type")
        try:
            if pd.notna(lo) and lo > 0:
                suffix = ""
                if stype == "hourly":  suffix = "/hr"
                elif stype == "monthly": suffix = "/mo"
                elif stype == "weekly":  suffix = "/wk"

                if stype in ("hourly", "monthly", "weekly"):
                    fmt = f"${lo:,.2f}" if lo < 100 else f"${lo:,.0f}"
                    if pd.notna(hi) and hi > lo:
                        fmt_hi = f"${hi:,.2f}" if hi < 100 else f"${hi:,.0f}"
                        return f"⚠️ {fmt} – {fmt_hi}{suffix}"
                    return f"⚠️ {fmt}{suffix}"
                else:
                    if pd.notna(hi) and hi > lo:
                        return f"${lo:,.0f} – ${hi:,.0f}"
                    return f"${lo:,.0f}"
        except Exception:
            pass
        return txt if isinstance(txt, str) else ""

    f2["Salary"] = f2.apply(fmt_salary, axis=1)

    # Format city population for display
    def fmt_pop(row):
        pop = row.get("city_pop")
        size = row.get("city_size", "")
        if pd.notna(pop) and pop > 0:
            return f"{int(pop):,} ({size})"
        return size if size and size != "Unknown" else ""
    f2["City Population"] = f2.apply(fmt_pop, axis=1)

    show_cols = ["title", "institution", "location_raw", "state", "r_tier_f",
                 "Salary", "City Population", "posted_ago", "published_date", "source", "link"]
    show_cols = [c for c in show_cols if c in f2.columns]

    rename_map = {
        "title": "Job Title", "institution": "Institution",
        "location_raw": "Location", "state": "State",
        "r_tier_f": "R-Tier", "posted_ago": "Posted",
        "published_date": "Date Posted", "source": "Source",
        "link": "Apply",
    }

    _NON_TENURE_PHRASES = ["non-tenure", "non tenure"]

    def role_bucket(row) -> str:
        t = (row.get("title") or "").lower()
        s = (row.get("summary") or "").lower()
        combined = t + " " + s
        if any(p in combined for p in _NON_TENURE_PHRASES):
            return "Non-Tenure"
        if "chair" in t:
            return "Chair / Leadership"
        if any(k in t for k in ["postdoc", "post doc", "postdoctoral", "post-doctoral"]):
            return "Postdoc"
        if "professor" in t:
            return "Assistant Professor"
        return "Other"

    f2["role_bucket"] = f2.apply(role_bucket, axis=1)

    def build_display_frame(df_in):
        out = df_in[show_cols].rename(columns=rename_map).copy()
        out["_sort_date"] = pd.to_datetime(out["Date Posted"], errors="coerce")
        out = out.sort_values(by=["_sort_date"], ascending=False, na_position="last")
        return out.drop(columns=["_sort_date"])

    table_cfg = {
        "Apply": st.column_config.LinkColumn("Apply", display_text="🔗 Apply"),
        "Salary": st.column_config.TextColumn("Salary", width="medium"),
        "City Population": st.column_config.TextColumn("City Population", width="medium"),
        "Posted": st.column_config.TextColumn("Posted", width="small"),
        "R-Tier": st.column_config.TextColumn("R-Tier", width="small"),
    }

    ap_df = build_display_frame(f2[f2["role_bucket"] == "Assistant Professor"])
    postdoc_df = build_display_frame(f2[f2["role_bucket"] == "Postdoc"])
    chair_df = build_display_frame(f2[f2["role_bucket"] == "Chair / Leadership"])
    nontenure_df = build_display_frame(f2[f2["role_bucket"] == "Non-Tenure"])

    st.markdown(f'<div class="section-hdr">🎓 Assistant Professor positions ({len(ap_df)})</div>', unsafe_allow_html=True)
    st.dataframe(ap_df, use_container_width=True, hide_index=True, height=360, column_config=table_cfg)

    st.markdown(f'<div class="section-hdr">🔬 Postdoc positions ({len(postdoc_df)})</div>', unsafe_allow_html=True)
    st.dataframe(postdoc_df, use_container_width=True, hide_index=True, height=360, column_config=table_cfg)

    st.markdown(f'<div class="section-hdr">📋 Non-Tenure positions ({len(nontenure_df)})</div>', unsafe_allow_html=True)
    st.dataframe(nontenure_df, use_container_width=True, hide_index=True, height=360, column_config=table_cfg)

    st.markdown(f'<div class="section-hdr">🪑 Chair / Leadership positions ({len(chair_df)})</div>', unsafe_allow_html=True)
    st.dataframe(chair_df, use_container_width=True, hide_index=True, height=360, column_config=table_cfg)

    csv = pd.concat([ap_df, postdoc_df, nontenure_df, chair_df], ignore_index=True).drop(columns=["Apply"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "health_jobs_all.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    c1, c2 = st.columns(2)

    # ── Jobs by R-Tier ───────────────────────────────────────────────────────
    with c1:
        st.markdown('<div class="section-hdr">Jobs by Research Tier</div>', unsafe_allow_html=True)
        tier_counts = f["r_tier_f"].value_counts().reset_index()
        tier_counts.columns = ["R-Tier","Count"]
        color_map = {k: "#{:02x}{:02x}{:02x}".format(*v) for k,v in RTIER_COLOR.items()}
        fig_tier = px.bar(
            tier_counts, x="R-Tier", y="Count",
            color="R-Tier", color_discrete_map=color_map,
            text="Count", template="plotly_white",
        )
        fig_tier.update_traces(textposition="outside")
        fig_tier.update_layout(showlegend=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig_tier, use_container_width=True)

    # ── Jobs by State ────────────────────────────────────────────────────────
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
        st.info("Not enough salary data in current filter to plot a distribution. Most postings do not disclose salary.")
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

    # ── Jobs by City Size ──────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-hdr">Jobs by City Size</div>', unsafe_allow_html=True)
        cs_order = ["Major metro (1M+)", "Large city (500K-1M)",
                     "Mid-size city (100K-500K)", "Small city (50K-100K)",
                     "Town (< 50K)", "Unknown"]
        cs_counts = f["city_size"].value_counts().reindex(cs_order).fillna(0).astype(int).reset_index()
        cs_counts.columns = ["City Size", "Count"]
        cs_counts = cs_counts[cs_counts["Count"] > 0]
        fig_cs = px.bar(
            cs_counts, x="City Size", y="Count",
            color="Count", color_continuous_scale="Teal",
            text="Count", template="plotly_white",
        )
        fig_cs.update_traces(textposition="outside")
        fig_cs.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20),
                             xaxis_tickangle=-25)
        st.plotly_chart(fig_cs, use_container_width=True)
    with c4:
        st.markdown('<div class="section-hdr">City Size Breakdown</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            cs_counts, names="City Size", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig_pie.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── US Choropleth — jobs per state ───────────────────────────────────────
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
st.caption("Sources: HigherEdJobs RSS · Carnegie Classification (2021 basic codes) · U.S. Census Bureau 2024 Population Estimates · Geocoding via OpenStreetMap Nominatim · Salary best-effort from posting text.")
