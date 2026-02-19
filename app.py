"""
Agency 8 — Influencer Heat Map (Streamlit Web App)

Add one section per client, upload their gift app CSV + posted CSV, then generate the map.
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
    "Add a section for each client, upload their CSVs, then click **Generate Map**. "
    "Dots are sized and colored by volume — darker = more activity."
)
st.divider()

# ── Client sections ───────────────────────────────────────────────────────────────

n_clients = st.number_input(
    "How many clients are you uploading for?",
    min_value=1, max_value=20, value=1, step=1,
)

client_configs = []

for i in range(int(n_clients)):
    with st.expander(f"Client #{i + 1}", expanded=True):

        client_name = st.text_input(
            "Client name",
            key=f"client_name_{i}",
            placeholder="e.g. Facile",
        )

        col_gift, col_archive = st.columns(2)

        with col_gift:
            st.markdown("**Gift App CSV**")
            gift_file = st.file_uploader(
                "Upload gift app CSV",
                type="csv",
                key=f"gift_{i}",
                label_visibility="collapsed",
            )

        with col_archive:
            st.markdown("**Posted CSV (from Archive)**")
            archive_file = st.file_uploader(
                "Upload posted CSV",
                type="csv",
                key=f"archive_{i}",
                label_visibility="collapsed",
            )

        # Column pickers — shown only after files are uploaded
        gift_col_config    = None
        archive_col_config = None

        if gift_file:
            gift_df_raw = pd.read_csv(gift_file, dtype=str)
            gcols = list(gift_df_raw.columns)
            with st.expander("Gift app column settings", expanded=False):
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    ig_default = auto_detect(gcols, ["instagram", "ig handle", "ig_handle"])
                    ig_col = st.selectbox(
                        "Instagram handle",
                        options=["(none)"] + gcols,
                        index=gcols.index(ig_default) + 1 if ig_default in gcols else 0,
                        key=f"ig_col_{i}",
                    )
                with gc2:
                    tt_default = auto_detect(gcols, ["tiktok", "tt handle", "tt_handle"])
                    tt_col = st.selectbox(
                        "TikTok handle",
                        options=["(none)"] + gcols,
                        index=gcols.index(tt_default) + 1 if tt_default in gcols else 0,
                        key=f"tt_col_{i}",
                    )
                with gc3:
                    zip_default = auto_detect(gcols, ["zip", "postal", "postcode", "post code"])
                    zip_col = st.selectbox(
                        "Zip code",
                        options=gcols,
                        index=gcols.index(zip_default) if zip_default in gcols else 0,
                        key=f"zip_col_{i}",
                    )
            gift_col_config = {"df": gift_df_raw, "ig": ig_col, "tt": tt_col, "zip": zip_col}

        if archive_file:
            archive_df_raw = pd.read_csv(archive_file, dtype=str)
            acols = list(archive_df_raw.columns)
            with st.expander("Posted CSV column settings", expanded=False):
                h_default = auto_detect(
                    acols,
                    ["handle", "username", "user", "social profile", "profile", "social", "account", "creator"],
                )
                handle_col = st.selectbox(
                    "Handle column (who posted)",
                    options=acols,
                    index=acols.index(h_default) if h_default in acols else 0,
                    key=f"handle_col_{i}",
                )
            archive_col_config = {"df": archive_df_raw, "handle_col": handle_col}

        if client_name and gift_col_config and archive_col_config:
            client_configs.append({
                "client":         client_name,
                "gift_config":    gift_col_config,
                "archive_config": archive_col_config,
            })

# ── Generate ──────────────────────────────────────────────────────────────────────

st.divider()
generate = st.button("Generate Map", type="primary", use_container_width=True)

if generate:
    if not client_configs:
        st.error("Please fill in at least one client with both CSVs uploaded and a client name.")
        st.stop()

    with st.spinner("Processing data and geocoding zip codes..."):

        all_gifted  = []
        all_posted  = []
        total_unmatched = 0

        for cfg in client_configs:
            client      = cfg["client"]
            gift_cfg    = cfg["gift_config"]
            archive_cfg = cfg["archive_config"]

            # Build gift records + handle→zip lookup for this client
            gift_rows     = []
            handle_to_zip = {}

            for _, row in gift_cfg["df"].iterrows():
                ig  = normalize_handle(row[gift_cfg["ig"]]) if gift_cfg["ig"] != "(none)" else ""
                tt  = normalize_handle(row[gift_cfg["tt"]]) if gift_cfg["tt"] != "(none)" else ""
                raw = str(row.get(gift_cfg["zip"], "")).strip()
                zip_code = raw.zfill(5)[:5] if raw and raw.lower() != "nan" else ""
                if zip_code and (ig or tt):
                    gift_rows.append({"ig_handle": ig, "tt_handle": tt, "zip_code": zip_code})
                    for h in [ig, tt]:
                        if h:
                            handle_to_zip[h] = zip_code

            for r in gift_rows:
                all_gifted.append({
                    "handle":      r["ig_handle"] or r["tt_handle"],
                    "client":      client,
                    "zip_code":    r["zip_code"],
                    "type":        "Gifted",
                    "posts_total": "",
                })

            # Match posted users to zip codes for this client
            posts_col = next(
                (c for c in archive_cfg["df"].columns
                 if "posts total" in c.lower() or "total posts" in c.lower()),
                None,
            )
            for _, row in archive_cfg["df"].iterrows():
                h = normalize_handle(row[archive_cfg["handle_col"]])
                if not h:
                    continue
                z = handle_to_zip.get(h)
                if z:
                    all_posted.append({
                        "handle":      h,
                        "client":      client,
                        "zip_code":    z,
                        "type":        "Posted",
                        "posts_total": str(row.get(posts_col, "")).strip() if posts_col else "",
                    })
                else:
                    total_unmatched += 1

        all_records = pd.DataFrame(all_gifted + all_posted)
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
        zip_lookup   = geocode_zip_codes(agg["zip_code"].tolist())
        agg["lat"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lat"))
        agg["lon"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lon"))
        agg["place"] = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("place", "Unknown"))
        agg = agg.dropna(subset=["lat", "lon"])

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Gifted", len(all_gifted))
    c2.metric("Posted & Matched", len(all_posted))
    c3.metric("Unmatched (no gift on file)", total_unmatched)

    # Map
    fig = build_map(agg)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Blue dots** = gifted influencers &nbsp;|&nbsp; "
        "**Red dots** = influencers who posted &nbsp;|&nbsp; "
        "Darker color + larger dot = higher volume in that area."
    )

    st.download_button(
        label="Download map as standalone HTML",
        data=fig.to_html(include_plotlyjs="cdn"),
        file_name="agency8_heatmap.html",
        mime="text/html",
    )

