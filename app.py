"""
Agency 8 — Influencer Heat Map (Streamlit Web App)

Upload your gift app CSV and archive CSV(s) to generate an interactive heat map.
"""

import streamlit as st
import pandas as pd
import pgeocode
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agency 8 — Heat Map",
    layout="wide",
)

# ── Color scales ──────────────────────────────────────────────────────────────────

COLOR_SCALES = {
    "Gifted": [[0, "#c6dbef"], [0.5, "#2171b5"], [1, "#08306b"]],  # light → dark blue
    "Posted": [[0, "#fcbba1"], [0.5, "#de2d26"], [1, "#67000d"]],  # light → dark red
}

# ── Helpers ───────────────────────────────────────────────────────────────────────

def normalize_handle(h):
    if not h or not isinstance(h, str):
        return ""
    return h.strip().lstrip("@").lower()


def auto_detect(cols, hints):
    """Return the first column whose name contains any hint (case-insensitive)."""
    for col in cols:
        if any(h.lower() in col.lower() for h in hints):
            return col
    return cols[0]


@st.cache_resource(show_spinner="Loading zip code database...")
def get_geocoder():
    return pgeocode.Nominatim("us")


def geocode_zip_codes(zip_codes):
    nomi = get_geocoder()
    results = nomi.query_postal_code(list(set(zip_codes)))
    lookup = {}
    for _, row in results.iterrows():
        zc = str(row["postal_code"]).strip().zfill(5)
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            lookup[zc] = {
                "lat":   row["latitude"],
                "lon":   row["longitude"],
                "place": f"{row['place_name']}, {row['state_name']}",
            }
    return lookup


def build_map(agg):
    fig = go.Figure()
    clients    = sorted(agg["client"].unique())
    types      = sorted(agg["type"].unique())
    global_max = max(agg["count"].max(), 1)

    for type_ in types:
        for client in clients:
            subset = agg[(agg["type"] == type_) & (agg["client"] == client)]
            if subset.empty:
                continue

            def hover_text(r):
                lines = [
                    f"<b>{r['place']}</b>",
                    f"ZIP: {r['zip_code']}",
                    f"People: {r['count']}",
                    f"Type: {r['type']}",
                    f"Client: {r['client']}",
                ]
                if r.get("total_posts", 0):
                    lines.append(f"Total Posts: {int(r['total_posts'])}")
                lines.append(f"Handles: {r['sample_handles']}")
                return "<br>".join(lines)

            hover = subset.apply(hover_text, axis=1).tolist()
            sizes = subset["count"].apply(lambda x: min(8 + x * 5, 50)).tolist()

            fig.add_trace(go.Scattermapbox(
                lat=subset["lat"].tolist(),
                lon=subset["lon"].tolist(),
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=subset["count"].tolist(),
                    colorscale=COLOR_SCALES.get(type_, [[0, "#ccc"], [1, "#333"]]),
                    cmin=1,
                    cmax=global_max,
                    opacity=0.85,
                    sizemode="diameter",
                    showscale=False,
                ),
                text=hover,
                hoverinfo="text",
                name=f"{type_} — {client}",
            ))

    all_true = [True] * len(fig.data)
    buttons  = [dict(label="All", method="update", args=[{"visible": all_true}])]

    for client in clients:
        vis = [client in t.name for t in fig.data]
        buttons.append(dict(label=client, method="update", args=[{"visible": vis}]))

    for type_ in types:
        vis = [type_ in t.name for t in fig.data]
        buttons.append(dict(label=f"{type_} only", method="update", args=[{"visible": vis}]))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 38.5, "lon": -96},
        mapbox_zoom=3,
        height=650,
        margin=dict(t=60, b=0, l=0, r=0),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.1,
            yanchor="top",
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(size=13),
        )],
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
    )
    return fig


# ── App UI ────────────────────────────────────────────────────────────────────────

st.title("Agency 8 — Influencer Heat Map")
st.markdown(
    "Upload your CSVs, confirm the columns, and click **Generate Map**. "
    "Dots are sized and colored by volume — darker = more activity."
)
st.divider()

# ── Step 1: Gift app CSV ──────────────────────────────────────────────────────────

st.subheader("Step 1 — Gift App CSV")
gift_file = st.file_uploader("Upload your gift app CSV", type="csv", key="gift")

gift_config = None
if gift_file:
    gift_raw = pd.read_csv(gift_file, dtype=str)
    cols     = list(gift_raw.columns)

    c1, c2, c3 = st.columns(3)
    with c1:
        ig_default = auto_detect(cols, ["instagram", "ig handle", "ig_handle"])
        ig_col = st.selectbox(
            "Instagram handle column",
            options=["(none)"] + cols,
            index=cols.index(ig_default) + 1 if ig_default in cols else 0,
        )
    with c2:
        tt_default = auto_detect(cols, ["tiktok", "tt handle", "tt_handle"])
        tt_col = st.selectbox(
            "TikTok handle column",
            options=["(none)"] + cols,
            index=cols.index(tt_default) + 1 if tt_default in cols else 0,
        )
    with c3:
        zip_default = auto_detect(cols, ["zip", "postal", "postcode", "post code"])
        zip_col = st.selectbox(
            "Zip code column",
            options=cols,
            index=cols.index(zip_default) if zip_default in cols else 0,
        )

    gift_config = {"df": gift_raw, "ig": ig_col, "tt": tt_col, "zip": zip_col}

# ── Step 2: Archive CSVs ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Step 2 — Archive CSVs (who posted)")

n_archive = st.number_input(
    "How many Archive CSVs are you uploading?",
    min_value=1, max_value=10, value=1, step=1,
)

archive_configs = []
for i in range(int(n_archive)):
    with st.expander(f"Archive CSV #{i + 1}", expanded=True):
        af = st.file_uploader(f"Upload archive CSV #{i + 1}", type="csv", key=f"archive_{i}")
        client_name = st.text_input(f"Client name", key=f"client_{i}", placeholder="e.g. Facile")

        if af and client_name:
            af_df = pd.read_csv(af, dtype=str)
            af_cols = list(af_df.columns)
            h_default = auto_detect(
                af_cols,
                ["handle", "username", "user", "social profile", "profile", "social", "account", "creator"],
            )
            handle_col = st.selectbox(
                "Handle column (who posted)",
                options=af_cols,
                index=af_cols.index(h_default) if h_default in af_cols else 0,
                key=f"handle_col_{i}",
            )
            archive_configs.append({
                "df": af_df,
                "handle_col": handle_col,
                "client": client_name,
            })

# ── Step 3: Generate ─────────────────────────────────────────────────────────────

st.divider()
generate = st.button("Generate Map", type="primary", use_container_width=True)

if generate:
    if not gift_config:
        st.error("Please upload your gift app CSV first.")
        st.stop()

    with st.spinner("Processing data and geocoding zip codes..."):

        # Build gift records
        gift_rows = []
        for _, row in gift_config["df"].iterrows():
            ig  = normalize_handle(row[gift_config["ig"]]) if gift_config["ig"] != "(none)" else ""
            tt  = normalize_handle(row[gift_config["tt"]]) if gift_config["tt"] != "(none)" else ""
            raw = str(row.get(gift_config["zip"], "")).strip()
            zip_code = raw.zfill(5)[:5] if raw and raw.lower() != "nan" else ""
            if zip_code and (ig or tt):
                gift_rows.append({"ig_handle": ig, "tt_handle": tt, "zip_code": zip_code})

        gift_df = pd.DataFrame(gift_rows)

        # Handle → zip lookup
        handle_to_zip = {}
        for _, row in gift_df.iterrows():
            for h in [row["ig_handle"], row["tt_handle"]]:
                if h:
                    handle_to_zip[h] = row["zip_code"]

        # Build posted records
        posted_records, unmatched = [], 0
        for cfg in archive_configs:
            posts_col = next(
                (c for c in cfg["df"].columns if "posts total" in c.lower() or "total posts" in c.lower()),
                None,
            )
            for _, row in cfg["df"].iterrows():
                h = normalize_handle(row[cfg["handle_col"]])
                if not h:
                    continue
                z = handle_to_zip.get(h)
                if z:
                    posted_records.append({
                        "handle":      h,
                        "client":      cfg["client"],
                        "zip_code":    z,
                        "type":        "Posted",
                        "posts_total": str(row.get(posts_col, "")).strip() if posts_col else "",
                    })
                else:
                    unmatched += 1

        # Gifted records
        gifted_records = [
            {
                "handle":      r["ig_handle"] or r["tt_handle"],
                "client":      "All Clients",
                "zip_code":    r["zip_code"],
                "type":        "Gifted",
                "posts_total": "",
            }
            for _, r in gift_df.iterrows()
        ]

        all_records = pd.DataFrame(gifted_records + posted_records)
        if all_records.empty:
            st.error("No data to map. Check your CSV files and column selections.")
            st.stop()

        # Aggregate
        agg = (
            all_records
            .groupby(["zip_code", "type", "client"])
            .agg(
                count=("handle", "count"),
                sample_handles=("handle", lambda x: ", ".join(sorted(x.dropna().unique())[:5])),
                total_posts=("posts_total", lambda x: sum(
                    int(v) for v in x if str(v).strip().isdigit()
                )),
            )
            .reset_index()
        )

        # Geocode
        zip_lookup = geocode_zip_codes(agg["zip_code"].tolist())
        agg["lat"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lat"))
        agg["lon"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lon"))
        agg["place"] = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("place", "Unknown"))
        agg = agg.dropna(subset=["lat", "lon"])

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Gifted", len(gift_df))
    c2.metric("Posted & Matched", len(posted_records))
    c3.metric("Unmatched (no gift on file)", unmatched)

    # Map
    fig = build_map(agg)
    st.plotly_chart(fig, use_container_width=True)

    # Legend note
    st.caption(
        "**Blue dots** = gifted influencers &nbsp;|&nbsp; "
        "**Red dots** = influencers who posted &nbsp;|&nbsp; "
        "Darker color + larger dot = higher volume in that area."
    )

    # Download
    st.download_button(
        label="Download map as standalone HTML",
        data=fig.to_html(include_plotlyjs="cdn"),
        file_name="agency8_heatmap.html",
        mime="text/html",
    )
