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


_STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

def normalize_state(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s or s.lower() in ("nan", "n/a", ""):
        return None
    upper = s.upper()
    if upper in _STATE_ABBREV:
        return _STATE_ABBREV[upper]
    # Title-case it so "california" → "California"
    return s.title()


def auto_detect(cols, hints):
    for col in cols:
        if any(h.lower() in col.lower() for h in hints):
            return col
    return cols[0]


_GIFT_DATE_FMTS = [
    "%m/%d/%Y %H:%M:%S",  # 2/16/2024 0:00:00
    "%m/%d/%Y %H:%M",     # 2/16/2024 0:00
    "%m/%d/%Y",           # 2/16/2024
    "%Y-%m-%d %H:%M:%S",  # 2025-10-06 18:31:06
    "%Y-%m-%d %H:%M",     # 2025-10-06 18:31
    "%Y-%m-%d",           # 2025-10-06
    "%m/%d/%y",           # 3/1/26
    "%m/%d/%y %H:%M:%S",  # 3/1/26 0:00:00
]

def parse_gift_dates(series):
    """Parse a Series of mixed-format date strings robustly."""
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    remaining = series.copy()
    for fmt in _GIFT_DATE_FMTS:
        mask = result.isna() & remaining.notna()
        if not mask.any():
            break
        parsed = pd.to_datetime(remaining[mask], format=fmt, errors="coerce")
        result[mask] = parsed
    # Final fallback for anything still unparsed
    still_na = result.isna() & remaining.notna()
    if still_na.any():
        result[still_na] = pd.to_datetime(remaining[still_na], errors="coerce", format="mixed")
    return result


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
                gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                with gc1:
                    ig_default = auto_detect(gcols, ["instagram", "ig handle", "ig_handle", "handle"])
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
                    date_default = auto_detect(gcols, ["timestamp", "date", "ship", "sent", "created", "gift date"])
                    gift_date_col = st.selectbox("Gift date", ["(none)"] + gcols,
                                                 index=gcols.index(date_default) + 1 if date_default in gcols else 1,
                                                 key=f"gift_date_col_{i}")
                with gc5:
                    state_default = auto_detect(gcols, ["state", "province", "region"])
                    gift_state_col = st.selectbox("State column (optional)", ["(none)"] + gcols,
                                                  index=gcols.index(state_default) + 1 if state_default in gcols else 0,
                                                  key=f"gift_state_col_{i}")
            gift_year = None
            if gift_date_col != "(none)":
                gift_year = st.number_input(
                    "What year were these gifts sent? (needed when the date column only shows month/day, e.g. '2/16')",
                    min_value=2018, max_value=2030, value=2024, step=1,
                    key=f"gift_year_{i}",
                )
            gift_col_config = {"df": gift_df_raw, "ig": ig_col, "tt": tt_col, "zip": zip_col, "date": gift_date_col, "year": gift_year, "state_col": gift_state_col}

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
                sc1, sc2, sc3, sc4 = st.columns(4)
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
                    # Prioritise "Created at" — avoid "Paid at" which is often empty
                    sdate_default = auto_detect(scols, ["created at", "order date"])
                    if sdate_default == scols[0]:  # fallback: try broader hints
                        sdate_default = auto_detect(scols, ["created", "date"])
                    shopify_date_col = st.selectbox(
                        "Order date column (optional)", ["(none)"] + scols,
                        index=scols.index(sdate_default) + 1 if sdate_default in scols else 0,
                        key=f"shopify_date_{i}",
                    )
                with sc4:
                    stag_default = auto_detect(scols, ["tags", "tag"])
                    shopify_tag_col = st.selectbox(
                        "Tags column (optional)", ["(none)"] + scols,
                        index=scols.index(stag_default) + 1 if stag_default in scols else 0,
                        key=f"shopify_tag_col_{i}",
                    )
                shopify_exclude_tags = st.text_input(
                    "Exclude orders with these tags (comma-separated, e.g. A8, gifted)",
                    value="A8",
                    key=f"shopify_exclude_tags_{i}",
                )
            shopify_col_config = {"df": shopify_df_raw, "zip": shopify_zip_col, "rev": shopify_rev_col, "date": shopify_date_col, "tag_col": shopify_tag_col, "exclude_tags": shopify_exclude_tags}

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
                    # If date looks like M/D or MM/DD (no year), append the campaign year
                    if date_str and date_str.count("/") == 1 and gift_cfg.get("year"):
                        date_str = f"{date_str}/{int(gift_cfg['year'])}"
                    if zip_code and (ig or tt):
                        gift_rows.append({"ig_handle": ig, "tt_handle": tt, "zip_code": zip_code})
                        for h in [ig, tt]:
                            if h:
                                handle_to_zip[h] = zip_code
                        if date_str and date_str.lower() != "nan":
                            state_direct = ""
                            if gift_cfg.get("state_col") and gift_cfg["state_col"] != "(none)":
                                state_direct = str(row.get(gift_cfg["state_col"], "")).strip()
                            gift_events_list.append({"client": client, "zip_code": zip_code, "gift_date": date_str, "state_direct": state_direct})

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
                # Build set of tags to exclude (e.g. {"a8", "gifted"})
                exclude_tags = set()
                if shopify_cfg.get("exclude_tags"):
                    exclude_tags = {t.strip().lower() for t in shopify_cfg["exclude_tags"].split(",") if t.strip()}
                for idx, row in shopify_cfg["df"].iterrows():
                    # Skip gifted/agency orders based on tag filter
                    if exclude_tags and shopify_cfg.get("tag_col") and shopify_cfg["tag_col"] != "(none)":
                        row_tags = str(row.get(shopify_cfg["tag_col"], "")).lower()
                        if any(t in row_tags for t in exclude_tags):
                            continue
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
            # Use direct state column if provided, otherwise infer from zip code
            if "state_direct" in gev.columns and gev["state_direct"].str.strip().ne("").any():
                gev["state"] = gev["state_direct"].map(normalize_state)
                # Fall back to zip geocoding for any rows with blank state
                mask = gev["state"].isna()
                if mask.any():
                    gev.loc[mask, "state"] = gev.loc[mask, "zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state"))
            else:
                gev["state"] = gev["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state"))
            gev["gift_date"] = parse_gift_dates(gev["gift_date"])
            st.session_state["gift_events_df"] = gev.dropna(subset=["gift_date", "state"])
        else:
            st.session_state["gift_events_df"] = None

        # Build shopify events dataframe (with dates + geocoded state)
        if shopify_events_list:
            sev = pd.DataFrame(shopify_events_list)
            sev["state"] = sev["zip_code"].map(lambda z: zip_lookup.get(z, {}).get("state"))
            sev["order_date"] = pd.to_datetime(sev["order_date"], errors="coerce", utc=True).dt.tz_convert(None)
            st.session_state["shopify_events_df"] = sev.dropna(subset=["order_date", "state"])
        else:
            st.session_state["shopify_events_df"] = None

        st.session_state["agg"]                      = agg
        st.session_state["total_gifted"]             = len(all_gifted)
        st.session_state["total_posted"]             = len(all_posted)
        st.session_state["total_shopify"]            = len(all_shopify)
        st.session_state["total_unmatched"]          = total_unmatched
        st.session_state["debug_gift_raw_count"]     = len(gift_events_list)
        if gift_events_list:
            sample = gift_events_list[0]
            st.session_state["debug_gift_sample"] = f"zip={sample.get('zip_code')} | date='{sample.get('gift_date')}' | state_direct='{sample.get('state_direct')}'"
            if len(gift_events_list) > 2:
                s2 = gift_events_list[2]
                st.session_state["debug_gift_sample3"] = f"zip={s2.get('zip_code')} | date='{s2.get('gift_date')}' | state_direct='{s2.get('state_direct')}'"
            else:
                st.session_state["debug_gift_sample3"] = "only 2 rows"
        else:
            st.session_state["debug_gift_sample"] = "empty"
            st.session_state["debug_gift_sample3"] = "empty"

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

        missing = []
        if gift_events is None or (gift_events is not None and gift_events.empty):
            missing.append("**Gift App CSV** — no date rows were captured (check the Gift Date column is not '(none)')")
        if shopify_events is None or (shopify_events is not None and shopify_events.empty):
            missing.append("**Shopify CSV** — no date rows were captured (check the Order Date column is not '(none)')")

        if missing:
            st.warning("Gifting Impact needs date data from both CSVs. Missing:\n\n" + "\n\n".join(f"- {m}" for m in missing))
            st.info("Select the correct date columns in each CSV's column settings expander, then click **Generate Map** again.")
        else:
            st.markdown(
                "Shows monthly Shopify orders per state with a marker at the first gift date. "
                "Look for an uptick after the gifting line — that's your signal."
            )

            with st.expander("🔍 Data check — verify what's loaded", expanded=True):
                raw_count = st.session_state.get("debug_gift_raw_count", "n/a")
                sample1   = st.session_state.get("debug_gift_sample", "n/a")
                sample3   = st.session_state.get("debug_gift_sample3", "n/a")
                st.caption(f"Raw gift rows with a date (before state mapping): **{raw_count}**")
                st.caption(f"Row 1 sample: {sample1}")
                st.caption(f"Row 3 sample: {sample3}")
                st.write(f"**Shopify events loaded:** {len(shopify_events)} rows")
                st.write(f"**Gift events loaded:** {len(gift_events)} rows")
                if not shopify_events.empty:
                    st.write(f"**Shopify date range:** {shopify_events['order_date'].min().date()} → {shopify_events['order_date'].max().date()}")
                    shopify_by_state = shopify_events.groupby("state").size().reset_index(name="orders").sort_values("orders", ascending=False)
                    st.write("**Shopify orders by state:**")
                    st.dataframe(shopify_by_state, use_container_width=True, hide_index=True)
                if not gift_events.empty:
                    st.write(f"**Gift date range:** {gift_events['gift_date'].min().date()} → {gift_events['gift_date'].max().date()}")
                    gift_by_state = gift_events.groupby("state").size().reset_index(name="gift_rows").sort_values("gift_rows", ascending=False)
                    st.write("**Gift dated rows by state** (states NOT listed here will show the 'no gift dates' message):")
                    st.dataframe(gift_by_state, use_container_width=True, hide_index=True)
                else:
                    st.warning("No gift events with dates were loaded. Check that the Gift Date column is set to 'Timestamp' (not '(none)') in the Gift App CSV column settings, then click Generate Map again.")

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

            # Gifted states — use agg (already correctly geocoded, same source as the maps)
            gifted_states_agg = set(
                agg[(agg["type"] == "Gifted") & agg["client"].isin(selected_clients)]["state"].dropna().unique()
            )
            # First gift date per state from gift_events_df (if dates were available)
            first_gift_date_by_state = {}
            if not first_gift.empty:
                for _, r in first_gift.iterrows():
                    first_gift_date_by_state[r["state"]] = r["first_gift_date"]

            # State picker — all states that have Shopify orders
            shopify_states = sorted(sev["state"].dropna().unique())

            if not shopify_states:
                st.warning("No states found in Shopify orders. Check that the zip code and date columns are selected correctly.")
            else:
                col_state, col_tf = st.columns([3, 1])
                with col_state:
                    selected_state = st.selectbox("Select a state to view", shopify_states, key="impact_state")
                with col_tf:
                    timeframe_options = {"All time": None, "3 months": 3, "6 months": 6, "1 year": 12, "2 years": 24}
                    selected_tf = st.selectbox("Timeframe", list(timeframe_options.keys()), index=0, key="impact_timeframe")
                    tf_months = timeframe_options[selected_tf]

                state_orders = sev[sev["state"] == selected_state].copy()
                if tf_months:
                    cutoff = pd.Timestamp.now() - pd.DateOffset(months=tf_months)
                    state_orders = state_orders[state_orders["order_date"] >= cutoff]
                state_orders["month"] = state_orders["order_date"].dt.strftime("%Y-%m")
                monthly = state_orders.groupby("month").size().reset_index(name="orders").sort_values("month")

                # Build full month range for selected timeframe
                if tf_months:
                    range_start = pd.Timestamp.now() - pd.DateOffset(months=tf_months)
                else:
                    earliest_gift_ts = min(first_gift_date_by_state.values(), default=None)
                    earliest_order_ts = sev["order_date"].min() if not sev.empty else None
                    range_start = min(
                        [d for d in [earliest_gift_ts, earliest_order_ts] if d is not None],
                        default=pd.Timestamp.now() - pd.DateOffset(years=1),
                    )
                full_months = [
                    d.strftime("%Y-%m")
                    for d in pd.date_range(start=range_start.strftime("%Y-%m"), end=pd.Timestamp.now(), freq="MS")
                ]
                monthly_full = (
                    pd.DataFrame({"month": full_months})
                    .merge(monthly, on="month", how="left")
                    .fillna(0)
                )

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=monthly_full["month"].tolist(),
                    y=monthly_full["orders"].tolist(),
                    name="Monthly Orders",
                    marker_color=SHOPIFY_COLOR,
                ))

                # Add first gift date line using add_shape (works on category axis)
                if selected_state in gifted_states_agg:
                    gift_date = first_gift_date_by_state.get(selected_state)
                    if gift_date:
                        gift_month_str = gift_date.strftime("%Y-%m")
                        if gift_month_str in monthly_full["month"].values:
                            fig_trend.add_shape(
                                type="line",
                                x0=gift_month_str, x1=gift_month_str,
                                y0=0, y1=1, yref="paper",
                                line=dict(dash="dash", color="white", width=2),
                            )
                            fig_trend.add_annotation(
                                x=gift_month_str, y=0.97, yref="paper",
                                text=f"First gift: {gift_date.strftime('%b %Y')}",
                                showarrow=False, xanchor="left",
                                font=dict(color="white", size=11),
                            )
                    else:
                        st.caption(f"ℹ️ {selected_state} was gifted but no dated gift rows were found for this state — some rows may have blank dates in the Gift App CSV.")
                else:
                    st.caption(f"ℹ️ No gifting recorded in {selected_state} yet — this is an untapped market.")

                fig_trend.update_layout(
                    height=420,
                    margin=dict(t=30, b=0, l=0, r=0),
                    xaxis_title="Month",
                    yaxis_title="Orders",
                    xaxis=dict(type="category"),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                # Summary table: all states, orders before vs after first gift
                merged = sev.merge(first_gift, on=["client", "state"], how="inner")
                merged["days_from_gift"] = (merged["order_date"] - merged["first_gift_date"]).dt.days
                before_all = merged[merged["days_from_gift"] < 0]
                after_all  = merged[merged["days_from_gift"] >= 0]

                before_stats = before_all.groupby(["client", "state"]).size().reset_index(name="orders_before")
                after_stats  = after_all.groupby(["client", "state"]).size().reset_index(name="orders_after")

                impact = first_gift.merge(before_stats, on=["client", "state"], how="left")
                impact = impact.merge(after_stats, on=["client", "state"], how="left").fillna(0)
                impact["change"] = (
                    (impact["orders_after"] - impact["orders_before"])
                    / impact["orders_before"].replace(0, float("nan")) * 100
                ).round(1)
                impact = impact.sort_values("orders_after", ascending=False).reset_index(drop=True)
                impact["first_gift_date"] = impact["first_gift_date"].dt.strftime("%Y-%m-%d")
                impact.index += 1

                st.subheader("All States — Orders Before vs After First Gift")
                display = impact[["client", "state", "first_gift_date", "orders_before", "orders_after", "change"]].copy()
                display.columns = ["Client", "State", "First Gift Date", "Orders Before Gifting", "Orders After Gifting", "Change %"]
                st.dataframe(
                    display.style.format({
                        "Orders Before Gifting": "{:.0f}",
                        "Orders After Gifting": "{:.0f}",
                        "Change %": "{:+.1f}%",
                    }),
                    use_container_width=True,
                )
                st.caption(
                    "⚠️ Correlation, not causation — order changes after gifting may reflect "
                    "seasonality, ads, or other factors. Use as a directional signal."
                )
