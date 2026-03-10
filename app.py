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

MARKER_BASE_SIZE = 10   # same for both Gifted and Posted
MARKER_SCALE     = 2.5  # grows with count
MARKER_OPACITY   = 0.90

SHOPIFY_COLOR = "#9b59b6"  # medium purple — distinct from all client colors


def neon_version(hex_color):
    """Return a neon/electric variant of a hex color — same hue, max saturation, higher lightness."""
    hex_color = hex_color.lstrip("#")
    r, g, b   = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
    h, l, s   = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, 0.72, 1.0)
    return "#{:02x}{:02x}{:02x}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def generate_client_colors(clients):
    """Generate one visually distinct color per client, no limit on count."""
    n = max(len(clients), 1)
    colors = {}
    for i, client in enumerate(sorted(clients)):
        hue        = i / n                          # evenly spaced around color wheel
        lightness  = 0.45                           # darker base so neon Posted dots pop
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

    # Dots only — no heat layer. Size scales with count so density is still visible.
    for type_ in types:
        for client in sorted(agg["client"].unique()):
            subset = agg[(agg["type"] == type_) & (agg["client"] == client)]
            if subset.empty:
                continue

            def hover_text(r):
                if r["type"] == "Shopify Customers":
                    lines = [
                        f"<b>{r['place']}</b>",
                        f"ZIP: {r['zip_code']}",
                        f"Customers: {r['count']}",
                        f"Client: {r['client']}",
                    ]
                    if r.get("revenue", 0):
                        lines.append(f"Revenue: ${r['revenue']:,.2f}")
                else:
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

            base_color = client_colors.get(client, "#888888")
            if type_ == "Shopify Customers":
                dot_color = SHOPIFY_COLOR
            elif type_ == "Gifted":
                dot_color = base_color
            else:
                dot_color = neon_version(base_color)
            sizes = subset["count"].apply(
                lambda x: min(MARKER_BASE_SIZE + x * MARKER_SCALE, MARKER_BASE_SIZE * 3)
            ).tolist()

            fig.add_trace(go.Scattermapbox(
                lat=subset["lat"].tolist(),
                lon=subset["lon"].tolist(),
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=dot_color,
                    opacity=MARKER_OPACITY,
                    sizemode="diameter",
                ),
                text=subset.apply(hover_text, axis=1).tolist(),
                hoverinfo="text",
                name=f"{client} ({type_})",
                showlegend=True,
            ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 38.5, "lon": -96},
        mapbox_zoom=3,
        height=620,
        margin=dict(t=10, b=0, l=0, r=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(color="#222", size=12),
            x=0.01, y=0.99,
            bordercolor="#ccc",
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
            title=dict(text="Post Rate %"),
            ticksuffix="%",
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
        mapbox_style="carto-positron",
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

        col_gift, col_archive, col_shopify = st.columns(3)
        with col_gift:
            st.markdown("**Gift App CSV**")
            gift_file = st.file_uploader("gift", type="csv", key=f"gift_{i}", label_visibility="collapsed")
        with col_archive:
            st.markdown("**Posted CSV (from Archive)**")
            archive_file = st.file_uploader("archive", type="csv", key=f"archive_{i}", label_visibility="collapsed")
        with col_shopify:
            st.markdown("**Shopify Customers CSV** *(optional)*")
            shopify_file = st.file_uploader("shopify", type="csv", key=f"shopify_{i}", label_visibility="collapsed")

        gift_col_config    = None
        archive_col_config = None
        shopify_col_config = None

        if gift_file:
            gift_df_raw = pd.read_csv(gift_file, dtype=str)
            gcols = list(gift_df_raw.columns)
            with st.expander("Gift app column settings", expanded=False):
                gc1, gc2, gc3, gc4 = st.columns(4)
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
                with gc4:
                    date_default = auto_detect(gcols, ["date", "ship", "sent", "created", "gift date"])
                    gift_date_col = st.selectbox("Gift date", ["(none)"] + gcols,
                                                 index=gcols.index(date_default) + 1 if date_default in gcols else 1,
                                                 key=f"gift_date_col_{i}")
            gift_col_config = {"df": gift_df_raw, "ig": ig_col, "tt": tt_col, "zip": zip_col, "date": gift_date_col}

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

        if shopify_file:
            shopify_df_raw = pd.read_csv(shopify_file, dtype=str)
            scols = list(shopify_df_raw.columns)
            with st.expander("Shopify column settings", expanded=False):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    sz_default = auto_detect(scols, ["zip", "shipping zip", "billing zip", "postal"])
                    shopify_zip_col = st.selectbox(
                        "Zip code column", scols,
                        index=scols.index(sz_default) if sz_default in scols else 0,
                        key=f"shopify_zip_{i}",
                    )
                with sc2:
                    rev_default = auto_detect(scols, ["total spent", "subtotal", "total", "revenue", "amount"])
                    shopify_rev_col = st.selectbox(
                        "Revenue column (optional)", ["(none)"] + scols,
                        index=scols.index(rev_default) + 1 if rev_default in scols else 0,
                        key=f"shopify_rev_{i}",
                    )
                with sc3:
                    sdate_default = auto_detect(scols, ["created at", "date", "order date", "paid at"])
                    shopify_date_col = st.selectbox(
                        "Order date column (optional)", ["(none)"] + scols,
                        index=scols.index(sdate_default) + 1 if sdate_default in scols else 0,
                        key=f"shopify_date_{i}",
                    )
            shopify_col_config = {"df": shopify_df_raw, "zip": shopify_zip_col, "rev": shopify_rev_col, "date": shopify_date_col}

        if client_name and (gift_col_config or shopify_col_config):
            client_configs.append({
                "client":         client_name,
                "gift_config":    gift_col_config,
                "archive_config": archive_col_config,
                "shopify_config": shopify_col_config,
            })

# ── Generate ──────────────────────────────────────────────────────────────────────

st.divider()
if st.button("Generate Map", type="primary", use_container_width=True):
    if not client_configs:
        st.error("Please fill in at least one client with both CSVs and a client name.")
        st.stop()

    with st.spinner("Processing data and geocoding zip codes..."):
        all_gifted, all_posted, all_shopify, total_unmatched = [], [], [], 0
        gift_events_list, shopify_events_list = [], []

        for cfg in client_configs:
            client      = cfg["client"]
            gift_cfg    = cfg.get("gift_config")
            archive_cfg = cfg.get("archive_config")
            shopify_cfg = cfg.get("shopify_config")

            handle_to_zip = {}

            if gift_cfg:
                gift_rows = []
                for _, row in gift_cfg["df"].iterrows():
                    ig  = normalize_handle(row[gift_cfg["ig"]]) if gift_cfg["ig"] != "(none)" else ""
                    tt  = normalize_handle(row[gift_cfg["tt"]]) if gift_cfg["tt"] != "(none)" else ""
                    raw = str(row.get(gift_cfg["zip"], "")).strip()
                    zip_code = raw.zfill(5)[:5] if raw and raw.lower() != "nan" else ""
                    date_str = str(row.get(gift_cfg["date"], "")).strip() if gift_cfg.get("date") and gift_cfg["date"] != "(none)" else ""
                    if zip_code and (ig or tt):
                        gift_rows.append({"ig_handle": ig, "tt_handle": tt, "zip_code": zip_code})
                        for h in [ig, tt]:
                            if h:
                                handle_to_zip[h] = zip_code
                        if date_str and date_str.lower() != "nan":
                            gift_events_list.append({"client": client, "zip_code": zip_code, "gift_date": date_str})

                for r in gift_rows:
                    all_gifted.append({
                        "handle": r["ig_handle"] or r["tt_handle"],
                        "client": client, "zip_code": r["zip_code"],
                        "type": "Gifted", "posts_total": "", "revenue": 0.0,
                    })

            if archive_cfg and gift_cfg:
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
                            "revenue": 0.0,
                        })
                    else:
                        total_unmatched += 1

            if shopify_cfg:
                for idx, row in shopify_cfg["df"].iterrows():
                    raw = str(row.get(shopify_cfg["zip"], "")).strip()
                    zip_code = raw.zfill(5)[:5] if raw and raw.lower() not in ("nan", "") else ""
                    if not zip_code:
                        continue
                    rev = 0.0
                    if shopify_cfg["rev"] != "(none)":
                        rev_str = str(row.get(shopify_cfg["rev"], "")).strip().replace("$", "").replace(",", "")
                        try:
                            rev = float(rev_str)
                        except ValueError:
                            rev = 0.0
                    all_shopify.append({
                        "handle": f"customer_{idx}",
                        "client": client, "zip_code": zip_code,
                        "type": "Shopify Customers", "posts_total": "", "revenue": rev,
                    })
                    order_date_str = str(row.get(shopify_cfg["date"], "")).strip() if shopify_cfg.get("date") and shopify_cfg["date"] != "(none)" else ""
                    if order_date_str and order_date_str.lower() != "nan":
                        shopify_events_list.append({"client": client, "zip_code": zip_code, "order_date": order_date_str, "revenue": rev})

        all_records = pd.DataFrame(all_gifted + all_posted + all_shopify)
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
                revenue=("revenue", "sum"),
            )
            .reset_index()
        )

        zip_lookup   = geocode_zip_codes(agg["zip_code"].tolist())
        agg["lat"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lat"))
        agg["lon"]   = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("lon"))
        agg["place"] = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("place", "Unknown"))
        agg["state"] = agg["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state", "Unknown"))
        agg = agg.dropna(subset=["lat", "lon"])

        # Build gift events dataframe (with dates + geocoded state)
        if gift_events_list:
            gev = pd.DataFrame(gift_events_list)
            gev["state"] = gev["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state"))
            gev["gift_date"] = pd.to_datetime(gev["gift_date"], errors="coerce")
            st.session_state["gift_events_df"] = gev.dropna(subset=["gift_date", "state"])
        else:
            st.session_state["gift_events_df"] = None

        # Build shopify events dataframe (with dates + geocoded state)
        if shopify_events_list:
            sev = pd.DataFrame(shopify_events_list)
            sev["state"] = sev["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state"))
            sev["order_date"] = pd.to_datetime(sev["order_date"], errors="coerce")
            st.session_state["shopify_events_df"] = sev.dropna(subset=["order_date", "state"])
        else:
            st.session_state["shopify_events_df"] = None

        st.session_state["agg"]             = agg
        st.session_state["total_gifted"]    = len(all_gifted)
        st.session_state["total_posted"]    = len(all_posted)
        st.session_state["total_shopify"]   = len(all_shopify)
        st.session_state["total_unmatched"] = total_unmatched

# ── Display ───────────────────────────────────────────────────────────────────────

if st.session_state["agg"] is not None:
    agg = st.session_state["agg"]

    # Assign a color to each client
    all_clients  = sorted(agg["client"].unique())
    client_colors = generate_client_colors(all_clients)

    st.divider()

    # ── Top metrics ───────────────────────────────────────────────────────────────
    total_gifted   = st.session_state.get("total_gifted", 0)
    total_posted   = st.session_state.get("total_posted", 0)
    total_shopify  = st.session_state.get("total_shopify", 0)
    post_rate_pct  = round(total_posted / total_gifted * 100, 1) if total_gifted else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Gifted",          f"{total_gifted:,}")
    m2.metric("Total Posted",          f"{total_posted:,}")
    m3.metric("Overall Post Rate",     f"{post_rate_pct}%")
    m4.metric("Shopify Customers",     f"{total_shopify:,}")

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
            neon = neon_version(color)
            legend_parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;">'
                f'<span style="width:12px;height:12px;border-radius:50%;'
                f'background:{color};display:inline-block;"></span>'
                f'<span style="width:12px;height:12px;border-radius:50%;'
                f'background:{neon};display:inline-block;"></span>'
                f'<b>{client}</b></span>'
            )
    shopify_legend = (
        '<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{SHOPIFY_COLOR};display:inline-block;"></span>'
        '<b>Shopify Customers</b></span>'
    ) if "Shopify Customers" in agg["type"].values else ""
    type_legend = (
        '<span style="opacity:0.7;font-size:13px;">'
        '&nbsp; Left dot = Gifted &nbsp;|&nbsp; Right dot (neon) = Posted'
        ' &nbsp;|&nbsp; Purple = Shopify Customers'
        ' &nbsp;|&nbsp; Bigger = more people in that area</span>'
    )
    st.markdown(
        f'<div style="padding:6px 0 10px 0;">'
        f'{"".join(legend_parts)}{shopify_legend}{type_legend}</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    tab_vol, tab_conv, tab_stats, tab_impact = st.tabs([
        "Volume Map", "Conversion Rate Map", "Stats & Leaderboards", "Gifting Impact"
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

        # ── Shopify Revenue by State ───────────────────────────────────────────────
        shopify_agg = agg[
            (agg["type"] == "Shopify Customers") & agg["client"].isin(selected_clients)
        ]
        if not shopify_agg.empty:
            st.divider()
            st.subheader("Shopify Customers by State")
            shopify_state = (
                shopify_agg.groupby("state")
                .agg(customers=("count", "sum"), revenue=("revenue", "sum"))
                .reset_index()
                .sort_values("customers", ascending=False)
                .reset_index(drop=True)
            )
            shopify_state.index += 1
            has_revenue = shopify_state["revenue"].sum() > 0
            if has_revenue:
                shopify_state.columns = ["State", "Customers", "Revenue ($)"]
                st.dataframe(
                    shopify_state.style.format({"Customers": "{:.0f}", "Revenue ($)": "${:,.2f}"}),
                    use_container_width=True,
                )
            else:
                shopify_state = shopify_state[["state", "customers"]]
                shopify_state.columns = ["State", "Customers"]
                st.dataframe(
                    shopify_state.style.format({"Customers": "{:.0f}"}),
                    use_container_width=True,
                )

            st.divider()

            # ── Opportunity Zones ──────────────────────────────────────────────────
            st.subheader("Opportunity Zones")
            st.caption("States with the most Shopify customers relative to gifting — where you should be doing more outreach.")
            gifted_agg = agg[(agg["type"] == "Gifted") & agg["client"].isin(selected_clients)]
            gifted_by_state = gifted_agg.groupby("state")["count"].sum().reset_index(name="gifted")
            shopify_by_state = shopify_agg.groupby("state")["count"].sum().reset_index(name="shopify_customers")
            gap = shopify_by_state.merge(gifted_by_state, on="state", how="left").fillna(0)
            gap["customers_per_gift"] = (
                gap["shopify_customers"] / gap["gifted"].replace(0, float("nan"))
            ).round(1)
            gap = gap.sort_values("customers_per_gift", ascending=False).reset_index(drop=True)
            gap.index += 1
            gap.columns = ["State", "Shopify Customers", "Gifted", "Customers per Gift"]
            st.dataframe(
                gap.style.format({
                    "Shopify Customers": "{:.0f}",
                    "Gifted": "{:.0f}",
                    "Customers per Gift": "{:.1f}",
                }),
                use_container_width=True,
            )

            st.divider()

            # ── Gifted vs Shopify Customers bar chart ─────────────────────────────
            st.subheader("Gifted vs Shopify Customers by State")
            st.caption("Side-by-side comparison — shows where gifting activity aligns with (or lags behind) your customer base.")
            chart_data = gap.copy().head(25).sort_values("Shopify Customers", ascending=False)
            fig_bar = go.Figure(data=[
                go.Bar(
                    name="Gifted",
                    x=chart_data["State"].tolist(),
                    y=chart_data["Gifted"].tolist(),
                    marker_color="#4a90d9",
                ),
                go.Bar(
                    name="Shopify Customers",
                    x=chart_data["State"].tolist(),
                    y=chart_data["Shopify Customers"].tolist(),
                    marker_color=SHOPIFY_COLOR,
                ),
            ])
            fig_bar.update_layout(
                barmode="group",
                height=420,
                margin=dict(t=10, b=0, l=0, r=0),
                legend=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(color="#222", size=12)),
                xaxis_title="State",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 4: Gifting Impact ──────────────────────────────────────────────────────
    with tab_impact:
        gift_events   = st.session_state.get("gift_events_df")
        shopify_events = st.session_state.get("shopify_events_df")

        if gift_events is None or shopify_events is None or gift_events.empty or shopify_events.empty:
            st.info(
                "To use this tab, make sure your **Gift App CSV** has a date column selected "
                "and your **Shopify CSV** has an order date column selected, then click Generate Map again."
            )
        else:
            st.markdown(
                "Shows how Shopify sales in each state changed after your first gift was sent there. "
                "Helps identify whether gifting is moving the needle on real customer sales."
            )

            window = st.selectbox("Days before/after first gift to compare", [30, 60, 90], index=1, key="impact_window")

            # Filter to selected clients
            gev = gift_events[gift_events["client"].isin(selected_clients)].copy()
            sev = shopify_events[shopify_events["client"].isin(selected_clients)].copy()

            # First gift date per (client, state)
            first_gift = (
                gev.groupby(["client", "state"])["gift_date"]
                .min()
                .reset_index()
                .rename(columns={"gift_date": "first_gift_date"})
            )

            # Join shopify orders to first gift dates
            merged = sev.merge(first_gift, on=["client", "state"], how="inner")
            merged["days_from_gift"] = (merged["order_date"] - merged["first_gift_date"]).dt.days

            before = merged[merged["days_from_gift"].between(-window, -1)]
            after  = merged[merged["days_from_gift"].between(0, window - 1)]

            before_stats = before.groupby(["client", "state"]).agg(
                orders_before=("revenue", "count"),
                revenue_before=("revenue", "sum"),
            ).reset_index()

            after_stats = after.groupby(["client", "state"]).agg(
                orders_after=("revenue", "count"),
                revenue_after=("revenue", "sum"),
            ).reset_index()

            impact = first_gift.merge(before_stats, on=["client", "state"], how="left")
            impact = impact.merge(after_stats, on=["client", "state"], how="left").fillna(0)
            impact["revenue_change_pct"] = (
                (impact["revenue_after"] - impact["revenue_before"])
                / impact["revenue_before"].replace(0, float("nan")) * 100
            ).round(1)
            impact = impact.sort_values("revenue_after", ascending=False).reset_index(drop=True)
            impact["first_gift_date"] = impact["first_gift_date"].dt.strftime("%Y-%m-%d")
            impact.index += 1

            if impact.empty:
                st.warning("Not enough overlapping data between gift dates and Shopify order dates to build this view.")
            else:
                # Summary bar chart: revenue before vs after by state
                top = impact.head(20)
                fig_impact = go.Figure(data=[
                    go.Bar(
                        name=f"Revenue {window}d Before Gifting",
                        x=top["state"].tolist(),
                        y=top["revenue_before"].tolist(),
                        marker_color="#aaaaaa",
                    ),
                    go.Bar(
                        name=f"Revenue {window}d After Gifting",
                        x=top["state"].tolist(),
                        y=top["revenue_after"].tolist(),
                        marker_color=SHOPIFY_COLOR,
                    ),
                ])
                fig_impact.update_layout(
                    barmode="group",
                    height=400,
                    margin=dict(t=10, b=0, l=0, r=0),
                    legend=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(color="#222", size=12)),
                    xaxis_title="State",
                    yaxis_title="Revenue ($)",
                )
                st.plotly_chart(fig_impact, use_container_width=True)

                st.subheader("Before vs After Gifting by State")
                display = impact[["client", "state", "first_gift_date",
                                   "orders_before", "revenue_before",
                                   "orders_after", "revenue_after",
                                   "revenue_change_pct"]].copy()
                display.columns = [
                    "Client", "State", "First Gift Date",
                    f"Orders ({window}d Before)", f"Revenue ({window}d Before)",
                    f"Orders ({window}d After)", f"Revenue ({window}d After)",
                    "Revenue Change %",
                ]
                st.dataframe(
                    display.style.format({
                        f"Orders ({window}d Before)": "{:.0f}",
                        f"Revenue ({window}d Before)": "${:,.2f}",
                        f"Orders ({window}d After)": "{:.0f}",
                        f"Revenue ({window}d After)": "${:,.2f}",
                        "Revenue Change %": "{:+.1f}%",
                    }),
                    use_container_width=True,
                )
                st.caption(
                    f"⚠️ Correlation, not causation — revenue changes in the {window} days after gifting "
                    "may reflect seasonality, ads, or other factors. Use as a directional signal."
                )

