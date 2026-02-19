"""
Agency 8 — Influencer Heat Map (Streamlit Web App)
"""

import colorsys
import streamlit as st
import pandas as pd
import pgeocode
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Agency 8 — Heat Map", layout="wide")

# ── Color config ──────────────────────────────────────────────────────────────────

# Heat map gradients (dark → bright = low → high density)
HEAT_SCALES = {
    "Gifted": [[0, "#000033"], [0.3, "#003399"], [0.65, "#0099ff"], [1.0, "#66ffff"]],
    "Posted": [[0, "#1a0000"], [0.3, "#990000"], [0.65, "#ff4400"], [1.0, "#ffee00"]],
}

# Marker symbols: solid circle = Gifted, hollow circle = Posted (same size)
TYPE_SYMBOL = {
    "Gifted": "circle",
    "Posted": "circle-stroked",
}
MARKER_SIZE = 11  # same size for both types


def generate_client_colors(clients):
    """Generate one visually distinct color per client, no limit on count."""
    n = max(len(clients), 1)
    colors = {}
    for i, client in enumerate(sorted(clients)):
        hue        = i / n                          # evenly spaced around color wheel
        lightness  = 0.62                           # bright enough to see on dark background
        saturation = 0.88
        r, g, b    = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors[client] = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
    return colors

# Conversion map: red = low post rate, green = high
CONVERSION_SCALE = [
    [0.0,  "#67000d"],
    [0.25, "#d73027"],
    [0.5,  "#fee08b"],
    [0.75, "#1a9850"],
    [1.0,  "#00441b"],
]

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
                "state": str(row["state_name"]),
            }
    return lookup


# ── Map builders ──────────────────────────────────────────────────────────────────

def build_volume_map(agg, client_colors):
    fig = go.Figure()
    types = sorted(agg["type"].unique())

    # Layer 1: heat blobs by type
    for type_ in types:
        subset_type = agg[agg["type"] == type_]
        if subset_type.empty:
            continue
        fig.add_trace(go.Densitymapbox(
            lat=subset_type["lat"].tolist(),
            lon=subset_type["lon"].tolist(),
            z=subset_type["count"].tolist(),
            radius=28,
            colorscale=HEAT_SCALES.get(type_, [[0, "#000"], [1, "#fff"]]),
            showscale=False,
            opacity=0.70,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Layer 2: markers colored by CLIENT, shape by TYPE (square=Gifted, circle=Posted)
    for type_ in types:
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
                    size=MARKER_SIZE,
                    color=client_colors.get(client, "#ffffff"),
                    symbol=TYPE_SYMBOL.get(type_, "circle"),
                    opacity=0.85,
                ),
                text=subset.apply(hover_text, axis=1).tolist(),
                hoverinfo="text",
                name=f"{client} ({type_})",
                showlegend=True,
            ))

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 38.5, "lon": -96},
        mapbox_zoom=3,
        height=620,
        margin=dict(t=10, b=0, l=0, r=0),
        legend=dict(
            bgcolor="rgba(0,0,0,0.6)",
            font=dict(color="white", size=12),
            x=0.01, y=0.99,
            bordercolor="#444",
            borderwidth=1,
        ),
    )
    return fig


def build_conversion_map(zip_stats):
    """Heat map where color = post rate % (red=low, green=high)."""
    fig = go.Figure()

    # Only include zips where at least 1 gift was sent
    data = zip_stats[zip_stats["gifted"] > 0].copy()
    if data.empty:
        return None

    def hover(r):
        return (
            f"<b>{r['place']}</b><br>"
            f"ZIP: {r['zip_code']}<br>"
            f"Gifted: {int(r['gifted'])}<br>"
            f"Posted: {int(r['posted'])}<br>"
            f"Post Rate: {r['post_rate']:.0f}%"
        )

    fig.add_trace(go.Densitymapbox(
        lat=data["lat"].tolist(),
        lon=data["lon"].tolist(),
        z=data["post_rate"].tolist(),
        radius=28,
        colorscale=CONVERSION_SCALE,
        showscale=True,
        colorbar=dict(
            title="Post Rate %",
            ticksuffix="%",
            bgcolor="rgba(0,0,0,0.5)",
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
        ),
        opacity=0.75,
        showlegend=False,
        hoverinfo="skip",
    ))

    # Dot overlay for hover
    fig.add_trace(go.Scattermapbox(
        lat=data["lat"].tolist(),
        lon=data["lon"].tolist(),
        mode="markers",
        marker=dict(size=7, color="white", opacity=0.0),
        text=data.apply(hover, axis=1).tolist(),
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 38.5, "lon": -96},
        mapbox_zoom=3,
        height=620,
        margin=dict(t=10, b=0, l=0, r=0),
    )
    return fig


# ── Analytics helpers ─────────────────────────────────────────────────────────────

def build_zip_stats(agg):
    gifted = agg[agg["type"] == "Gifted"].groupby("zip_code")["count"].sum().reset_index(name="gifted")
    posted = agg[agg["type"] == "Posted"].groupby("zip_code")["count"].sum().reset_index(name="posted")
    stats  = gifted.merge(posted, on="zip_code", how="left").fillna(0)
    stats["post_rate"] = (stats["posted"] / stats["gifted"].replace(0, float("nan")) * 100).round(1)
    geo = agg[["zip_code", "lat", "lon", "place", "state"]].drop_duplicates("zip_code")
    return stats.merge(geo, on="zip_code", how="left")


def build_state_stats(zip_stats):
    state = (
        zip_stats.groupby("state")
        .agg(gifted=("gifted", "sum"), posted=("posted", "sum"))
        .reset_index()
    )
    state["post_rate"] = (state["posted"] / state["gifted"].replace(0, float("nan")) * 100).round(1)
    return state.sort_values("post_rate", ascending=False)


def build_client_stats(agg):
    gifted = agg[agg["type"] == "Gifted"].groupby("client")["count"].sum().reset_index(name="gifted")
    posted = agg[agg["type"] == "Posted"].groupby("client")["count"].sum().reset_index(name="posted")
    stats  = gifted.merge(posted, on="client", how="left").fillna(0)
    stats["post_rate"] = (stats["posted"] / stats["gifted"].replace(0, float("nan")) * 100).round(1)
    return stats.sort_values("post_rate", ascending=False)


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
            gift_file = st.file_uploader("gift", type="csv", key=f"gift_{i}", label_visibility="collapsed")
        with col_archive:
            st.markdown("**Posted CSV (from Archive)**")
            archive_file = st.file_uploader("archive", type="csv", key=f"archive_{i}", label_visibility="collapsed")

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
        agg["state"] = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state", "Unknown"))
        agg = agg.dropna(subset=["lat", "lon"])

        st.session_state["agg"]             = agg
        st.session_state["total_gifted"]    = len(all_gifted)
        st.session_state["total_posted"]    = len(all_posted)
        st.session_state["total_unmatched"] = total_unmatched

# ── Display ───────────────────────────────────────────────────────────────────────

if st.session_state["agg"] is not None:
    agg = st.session_state["agg"]

    # Assign a color to each client
    all_clients  = sorted(agg["client"].unique())
    client_colors = generate_client_colors(all_clients)

    st.divider()

    # ── Top metrics ───────────────────────────────────────────────────────────────
    total_gifted  = st.session_state.get("total_gifted", 0)
    total_posted  = st.session_state.get("total_posted", 0)
    post_rate_pct = round(total_posted / total_gifted * 100, 1) if total_gifted else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Gifted",   f"{total_gifted:,}")
    m2.metric("Total Posted",   f"{total_posted:,}")
    m3.metric("Overall Post Rate", f"{post_rate_pct}%")

    st.markdown("#### Filters")
    f1, f2 = st.columns(2)
    with f1:
        selected_clients = st.multiselect(
            "Clients", options=all_clients, default=all_clients,
        )
    with f2:
        all_types = sorted(agg["type"].unique())
        selected_types = st.multiselect(
            "Type", options=all_types, default=all_types,
        )

    filtered = agg[agg["client"].isin(selected_clients) & agg["type"].isin(selected_types)]

    if filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # ── Client color legend ───────────────────────────────────────────────────────
    legend_parts = []
    for client, color in client_colors.items():
        if client in selected_clients:
            legend_parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;">'
                f'<span style="width:12px;height:12px;border-radius:50%;'
                f'background:{color};display:inline-block;"></span>'
                f'<b>{client}</b></span>'
            )
    type_legend = (
        '<span style="opacity:0.7;font-size:13px;">'
        '&nbsp; ● Solid = Gifted &nbsp;|&nbsp; ○ Hollow = Posted &nbsp;|&nbsp;'
        ' Heat glow: <span style="color:#66ffff">■</span> Gifted &nbsp;'
        '<span style="color:#ffaa00">■</span> Posted</span>'
    )
    st.markdown(
        f'<div style="padding:6px 0 10px 0;">'
        f'{"".join(legend_parts)}{type_legend}</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tab_vol, tab_conv, tab_stats = st.tabs([
        "Volume Map", "Conversion Rate Map", "Stats & Leaderboards"
    ])

    zip_stats   = build_zip_stats(agg)
    state_stats = build_state_stats(zip_stats)

    # ── Tab 1: Volume map ─────────────────────────────────────────────────────────
    with tab_vol:
        fig_vol = build_volume_map(filtered, client_colors)
        st.plotly_chart(fig_vol, use_container_width=True)
        st.download_button(
            "Download volume map as HTML",
            data=fig_vol.to_html(include_plotlyjs="cdn"),
            file_name="a8_volume_map.html", mime="text/html",
        )

    # ── Tab 2: Conversion rate map ────────────────────────────────────────────────
    with tab_conv:
        # Filter zip_stats to selected clients
        filtered_zip = build_zip_stats(
            agg[agg["client"].isin(selected_clients)]
        )
        fig_conv = build_conversion_map(filtered_zip)
        if fig_conv:
            st.caption(
                "Color = post rate (% of gifted who posted). "
                "**Red** = low conversion · **Green** = high conversion. "
                "Only shows zip codes where at least 1 gift was sent."
            )
            st.plotly_chart(fig_conv, use_container_width=True)
            st.download_button(
                "Download conversion map as HTML",
                data=fig_conv.to_html(include_plotlyjs="cdn"),
                file_name="a8_conversion_map.html", mime="text/html",
            )
        else:
            st.info("Not enough data to build conversion map.")

    # ── Tab 3: Stats & leaderboards ───────────────────────────────────────────────
    with tab_stats:

        # Per-client breakdown
        st.subheader("Per-Client Breakdown")
        client_stats = build_client_stats(agg[agg["client"].isin(selected_clients)])
        client_stats.columns = ["Client", "Gifts Sent", "Posts Received", "Post Rate %"]
        st.dataframe(
            client_stats.style.format({"Post Rate %": "{:.1f}%",
                                        "Gifts Sent": "{:.0f}",
                                        "Posts Received": "{:.0f}"}),
            use_container_width=True, hide_index=True,
        )

        st.divider()

        # State leaderboards
        col_top, col_bot = st.columns(2)

        with col_top:
            st.subheader("Top 10 States — Best Post Rate")
            st.caption("States where gifting converts to posts most effectively.")
            top_states = (
                state_stats[state_stats["gifted"] >= 2]
                .head(10)
                .reset_index(drop=True)
            )
            top_states.index += 1
            top_states.columns = ["State", "Gifts", "Posts", "Post Rate %"]
            st.dataframe(
                top_states.style.format({"Post Rate %": "{:.1f}%",
                                          "Gifts": "{:.0f}",
                                          "Posts": "{:.0f}"}),
                use_container_width=True,
            )

        with col_bot:
            st.subheader("Dead Zones — Most Gifts, Lowest Return")
            st.caption("States with the most gifts sent but lowest post rate.")
            dead_zones = (
                state_stats[state_stats["gifted"] >= 2]
                .sort_values(["gifted", "post_rate"], ascending=[False, True])
                .head(10)
                .reset_index(drop=True)
            )
            dead_zones.index += 1
            dead_zones.columns = ["State", "Gifts", "Posts", "Post Rate %"]
            st.dataframe(
                dead_zones.style.format({"Post Rate %": "{:.1f}%",
                                          "Gifts": "{:.0f}",
                                          "Posts": "{:.0f}"}),
                use_container_width=True,
            )

        st.divider()

        st.subheader("All States")
        all_states = state_stats.reset_index(drop=True)
        all_states.index += 1
        all_states.columns = ["State", "Gifts", "Posts", "Post Rate %"]
        st.dataframe(
            all_states.style.format({"Post Rate %": "{:.1f}%",
                                      "Gifts": "{:.0f}",
                                      "Posts": "{:.0f}"}),
            use_container_width=True,
        )

