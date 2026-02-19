"""
Agency 8 — Influencer Heat Map (Streamlit Web App)
"""

import streamlit as st
import pandas as pd
import pgeocode
import plotly.graph_objects as go

st.set_page_config(page_title="Agency 8 — Heat Map", layout="wide")

# ── Color scales ──────────────────────────────────────────────────────────────────

COLOR_SCALES = {
    "Gifted": [[0, "#6baed6"], [1, "#08306b"]],  # medium → dark blue
    "Posted": [[0, "#fc8d59"], [1, "#67000d"]],  # medium orange → dark red
}

LEGEND_HTML = """
<div style="display:flex; gap:28px; align-items:center; padding:8px 0 4px 0; font-size:14px;">
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="display:flex; gap:3px; align-items:center;">
            <div style="width:12px;height:12px;border-radius:50%;background:#6baed6;opacity:0.9"></div>
            <div style="width:16px;height:16px;border-radius:50%;background:#2171b5;opacity:0.9"></div>
            <div style="width:20px;height:20px;border-radius:50%;background:#08306b;opacity:0.9"></div>
        </div>
        <span><b>Gifted</b> — light = fewer, dark = more</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="display:flex; gap:3px; align-items:center;">
            <div style="width:12px;height:12px;border-radius:50%;background:#fc8d59;opacity:0.9"></div>
            <div style="width:16px;height:16px;border-radius:50%;background:#de2d26;opacity:0.9"></div>
            <div style="width:20px;height:20px;border-radius:50%;background:#67000d;opacity:0.9"></div>
        </div>
        <span><b>Posted</b> — light = fewer, dark = more</span>
    </div>
</div>
"""

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

    # Cap color scale at 75th percentile so gradient is visible even when
    # most zip codes have count=1
    color_max = max(float(agg["count"].quantile(0.75)), 2)

    for type_ in sorted(agg["type"].unique()):
        for client in sorted(agg["client"].unique()):
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

            fig.add_trace(go.Scattermapbox(
                lat=subset["lat"].tolist(),
                lon=subset["lon"].tolist(),
                mode="markers",
                marker=dict(
                    size=subset["count"].apply(lambda x: min(8 + x * 5, 50)).tolist(),
                    color=subset["count"].tolist(),
                    colorscale=COLOR_SCALES.get(type_, [[0, "#aaa"], [1, "#333"]]),
                    cmin=1,
                    cmax=color_max,
                    opacity=0.85,
                    sizemode="diameter",
                    showscale=False,
                ),
                text=subset.apply(hover_text, axis=1).tolist(),
                hoverinfo="text",
                name=f"{type_} — {client}",
                showlegend=False,
            ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 38.5, "lon": -96},
        mapbox_zoom=3,
        height=650,
        margin=dict(t=20, b=0, l=0, r=0),
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────────

if "agg" not in st.session_state:
    st.session_state["agg"] = None

# ── App UI ────────────────────────────────────────────────────────────────────────

st.title("Agency 8 — Influencer Heat Map")
st.markdown("Add a section per client, upload their CSVs, then click **Generate Map**.")
st.divider()

# ── Client sections ───────────────────────────────────────────────────────────────

n_clients = st.number_input(
    "How many clients are you uploading for?",
    min_value=1, max_value=20, value=1, step=1,
)

client_configs = []

for i in range(int(n_clients)):
    with st.expander(f"Client #{i + 1}", expanded=True):
        client_name = st.text_input("Client name", key=f"client_name_{i}", placeholder="e.g. Facile")

        col_gift, col_archive = st.columns(2)
        with col_gift:
            st.markdown("**Gift App CSV**")
            gift_file = st.file_uploader("Upload gift app CSV", type="csv", key=f"gift_{i}", label_visibility="collapsed")
        with col_archive:
            st.markdown("**Posted CSV (from Archive)**")
            archive_file = st.file_uploader("Upload posted CSV", type="csv", key=f"archive_{i}", label_visibility="collapsed")

        gift_col_config    = None
        archive_col_config = None

        if gift_file:
            gift_df_raw = pd.read_csv(gift_file, dtype=str)
            gcols = list(gift_df_raw.columns)
            with st.expander("Gift app column settings", expanded=False):
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    ig_default = auto_detect(gcols, ["instagram", "ig handle", "ig_handle"])
                    ig_col = st.selectbox("Instagram handle", ["(none)"] + gcols,
                                          index=gcols.index(ig_default) + 1 if ig_default in gcols else 0,
                                          key=f"ig_col_{i}")
                with gc2:
                    tt_default = auto_detect(gcols, ["tiktok", "tt handle", "tt_handle"])
                    tt_col = st.selectbox("TikTok handle", ["(none)"] + gcols,
                                          index=gcols.index(tt_default) + 1 if tt_default in gcols else 0,
                                          key=f"tt_col_{i}")
                with gc3:
                    zip_default = auto_detect(gcols, ["zip", "postal", "postcode", "post code"])
                    zip_col = st.selectbox("Zip code", gcols,
                                           index=gcols.index(zip_default) if zip_default in gcols else 0,
                                           key=f"zip_col_{i}")
            gift_col_config = {"df": gift_df_raw, "ig": ig_col, "tt": tt_col, "zip": zip_col}

        if archive_file:
            archive_df_raw = pd.read_csv(archive_file, dtype=str)
            acols = list(archive_df_raw.columns)
            with st.expander("Posted CSV column settings", expanded=False):
                h_default = auto_detect(acols, ["handle", "username", "user", "social profile",
                                                 "profile", "social", "account", "creator"])
                handle_col = st.selectbox("Handle column (who posted)", acols,
                                          index=acols.index(h_default) if h_default in acols else 0,
                                          key=f"handle_col_{i}")
            archive_col_config = {"df": archive_df_raw, "handle_col": handle_col}

        if client_name and gift_col_config and archive_col_config:
            client_configs.append({
                "client":         client_name,
                "gift_config":    gift_col_config,
                "archive_config": archive_col_config,
            })

# ── Generate ──────────────────────────────────────────────────────────────────────

st.divider()
if st.button("Generate Map", type="primary", use_container_width=True):
    if not client_configs:
        st.error("Please fill in at least one client with both CSVs and a client name.")
        st.stop()

    with st.spinner("Processing data and geocoding zip codes..."):
        all_gifted, all_posted, total_unmatched = [], [], 0

        for cfg in client_configs:
            client      = cfg["client"]
            gift_cfg    = cfg["gift_config"]
            archive_cfg = cfg["archive_config"]

            gift_rows, handle_to_zip = [], {}
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
                    "handle": r["ig_handle"] or r["tt_handle"],
                    "client": client, "zip_code": r["zip_code"],
                    "type": "Gifted", "posts_total": "",
                })

            posts_col = next(
                (c for c in archive_cfg["df"].columns
                 if "posts total" in c.lower() or "total posts" in c.lower()), None,
            )
            for _, row in archive_cfg["df"].iterrows():
                h = normalize_handle(row[archive_cfg["handle_col"]])
                if not h:
                    continue
                z = handle_to_zip.get(h)
                if z:
                    all_posted.append({
                        "handle": h, "client": client, "zip_code": z,
                        "type": "Posted",
                        "posts_total": str(row.get(posts_col, "")).strip() if posts_col else "",
                    })
                else:
                    total_unmatched += 1

        all_records = pd.DataFrame(all_gifted + all_posted)
        if all_records.empty:
            st.error("No data to map.")
            st.stop()

        agg = (
            all_records
            .groupby(["zip_code", "type", "client"])
            .agg(
                count=("handle", "count"),
                sample_handles=("handle", lambda x: ", ".join(sorted(x.dropna().unique())[:5])),
                total_posts=("posts_total", lambda x: sum(int(v) for v in x if str(v).strip().isdigit())),
            )
            .reset_index()
        )

        zip_lookup   = geocode_zip_codes(agg["zip_code"].tolist())
        agg["lat"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lat"))
        agg["lon"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lon"))
        agg["place"] = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("place", "Unknown"))
        agg = agg.dropna(subset=["lat", "lon"])

        st.session_state["agg"]             = agg
        st.session_state["total_gifted"]    = len(all_gifted)
        st.session_state["total_posted"]    = len(all_posted)
        st.session_state["total_unmatched"] = total_unmatched

# ── Map display + filters ─────────────────────────────────────────────────────────

if st.session_state["agg"] is not None:
    agg = st.session_state["agg"]

    st.divider()

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Gifted",            st.session_state.get("total_gifted", "—"))
    c2.metric("Posted & Matched",        st.session_state.get("total_posted", "—"))
    c3.metric("Unmatched (not gifted)",  st.session_state.get("total_unmatched", "—"))

    st.markdown("#### Filters")
    f1, f2 = st.columns(2)
    with f1:
        all_clients = sorted(agg["client"].unique())
        selected_clients = st.multiselect(
            "Clients — select one or more",
            options=all_clients,
            default=all_clients,
        )
    with f2:
        all_types = sorted(agg["type"].unique())
        selected_types = st.multiselect(
            "Type — Gifted, Posted, or both",
            options=all_types,
            default=all_types,
        )

    filtered = agg[agg["client"].isin(selected_clients) & agg["type"].isin(selected_types)]

    if filtered.empty:
        st.warning("No data matches the current filters.")
    else:
        st.markdown(LEGEND_HTML, unsafe_allow_html=True)
        fig = build_map(filtered)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label="Download map as standalone HTML",
            data=fig.to_html(include_plotlyjs="cdn"),
            file_name="agency8_heatmap.html",
            mime="text/html",
        )

