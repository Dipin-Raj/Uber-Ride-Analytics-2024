# the main code

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from functools import lru_cache

st.set_page_config(page_title="Uber Ride Analytics 2024", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme
dark_theme_css = """
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    [data-testid="stMetric"] {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
        color: #FAFAFA;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        flex: 1; /* Add this line to make metrics take equal space */
    }
    [data-testid="stMetric"] > div {
        border-bottom: 2px solid #1E90FF;
    }
    [data-testid="stMetricLabel"] p {
        color: #A0A0A0;
    }
    .st-emotion-cache-1r4qj8v {
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 1rem;
        background-color: #1a1c22;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .st-emotion-cache-183lzff {
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FAFAFA;
    }
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Color palette 
current_palette = px.colors.qualitative.T10
template = "plotly_dark"



# Data Loading
@st.cache_data(show_spinner=True)
def load_main(path):
    df = pd.read_csv(path, low_memory=False)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # parse datetime if present
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except:
            # fallback: combine Date and Time columns if present
            if 'Date' in df.columns and 'Time' in df.columns:
                df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
            else:
                df['datetime'] = pd.to_datetime(df.get('Date', None), errors='coerce')
    else:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df.get('Date', None), errors='coerce')

    # Ensure numeric fields
    for col in ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create additional time features
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.strftime('%b')
    df['weekday'] = df['datetime'].dt.day_name()
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday'])
    if 'Booking Status' in df.columns:
        df['Booking Status'] = df['Booking Status'].astype(str).str.strip().str.title()
    return df

@st.cache_data(show_spinner=True)
def load_coords(path, pickup=True):
    # pickup/drop CSVs
    coords = pd.read_csv(path)
    coords.columns = [c.strip() for c in coords.columns]
    # unify to Location
    if 'Pickup Location' in coords.columns:
        coords = coords.rename(columns={'Pickup Location': 'Location'})
    elif 'Drop Location' in coords.columns:
        coords = coords.rename(columns={'Drop Location': 'Location'})
    elif 'Location' not in coords.columns:
        # fallback
        coords = coords.rename(columns={coords.columns[0]: 'Location'})
    # rename lat/lon columns
    lat_cols = [c for c in coords.columns if 'lat' in c.lower()]
    lon_cols = [c for c in coords.columns if 'lon' in c.lower()]
    if lat_cols:
        coords = coords.rename(columns={lat_cols[0]: 'Latitude'})
    if lon_cols:
        coords = coords.rename(columns={lon_cols[0]: 'Longitude'})
    coords['Location'] = coords['Location'].astype(str).str.strip()
    return coords[['Location', 'Latitude', 'Longitude']]

# Paths
MAIN_PATH = r"D:\OMNIe\Dashboard\Uber_DT_Sorted.csv"
PICKUP_COORDS_PATH = r"D:\OMNIe\Dashboard\pickup_location_coords_delhi.csv"
DROP_COORDS_PATH = r"D:\OMNIe\Dashboard\drop_location_coords_delhi.csv"

df = load_main(MAIN_PATH)
pickup_coords = load_coords(PICKUP_COORDS_PATH, pickup=True)
drop_coords = load_coords(DROP_COORDS_PATH, pickup=False)

# normalize pickup/drop names
if 'Pickup Location' in df.columns:
    df['Pickup Location'] = df['Pickup Location'].astype(str).str.strip()
if 'Drop Location' in df.columns:
    df['Drop Location'] = df['Drop Location'].astype(str).str.strip()

df = df.merge(pickup_coords.rename(columns={'Location': 'Pickup Location', 'Latitude':'Pickup_Latitude', 'Longitude':'Pickup_Longitude'}),
              on='Pickup Location', how='left')
df = df.merge(drop_coords.rename(columns={'Location': 'Drop Location', 'Latitude':'Drop_Latitude', 'Longitude':'Drop_Longitude'}),
              on='Drop Location', how='left')

# Sidebar - Filters
st.sidebar.title("Filters")
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

vehicle_types = ["All"] + (sorted(df['Vehicle Type'].dropna().unique().tolist()) if 'Vehicle Type' in df.columns else [])
selected_vehicles = st.sidebar.selectbox("Vehicle Type", options=vehicle_types, index=0)

statuses = ["All"] + (sorted(df['Booking Status'].dropna().unique().tolist()) if 'Booking Status' in df.columns else [])
selected_statuses = st.sidebar.selectbox("Booking Status", options=statuses, index=0)

all_pickup_locations = ["All"] + (sorted(df['Pickup Location'].dropna().unique().tolist()) if 'Pickup Location' in df.columns else [])
selected_pickup_location = st.sidebar.selectbox("Pickup Location", options=all_pickup_locations, index=0)

all_drop_locations = ["All"] + (sorted(df['Drop Location'].dropna().unique().tolist()) if 'Drop Location' in df.columns else [])
selected_drop_location = st.sidebar.selectbox("Drop Location", options=all_drop_locations, index=0)

min_booking_value = float(df['Booking Value'].min()) if 'Booking Value' in df.columns else 0.0
max_booking_value = float(df['Booking Value'].max()) if 'Booking Value' in df.columns else 1000.0
bv_range = st.sidebar.slider("Booking Value range", min_value=float(np.nan_to_num(min_booking_value)), max_value=float(np.nan_to_num(max_booking_value)), value=(float(np.nan_to_num(min_booking_value)), float(np.nan_to_num(max_booking_value))))

# Apply filters
mask = pd.Series(True, index=df.index) 

# Date filter
mask_date_range = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
if date_range[0] == min_date and date_range[1] == max_date:
    mask = mask & (mask_date_range | df['date'].isna())
else:
    mask = mask & mask_date_range

# Vehicle Type filter
if selected_vehicles != "All":
    mask = mask & df['Vehicle Type'].isin([selected_vehicles]) 
else: # If "All" is selected, include NaNs
    mask = mask & (df['Vehicle Type'].isna() | df['Vehicle Type'].notna()) # Effectively no filter

# Booking Status filter
if selected_statuses != "All": # Only filter if not "All" is selected
    mask = mask & df['Booking Status'].isin([selected_statuses]) # Use isin with a list for single selection
else: # If "All" is selected, include NaNs
    mask = mask & (df['Booking Status'].isna() | df['Booking Status'].notna()) # Effectively no filter

# Pickup Location filter
if selected_pickup_location != "All": # Only filter if not "All" is selected
    mask = mask & df['Pickup Location'].isin([selected_pickup_location])
else: # If "All" is selected, include NaNs
    mask = mask & (df['Pickup Location'].isna() | df['Pickup Location'].notna())

# Drop Location filter
if selected_drop_location != "All": # Only filter if not "All" is selected
    mask = mask & df['Drop Location'].isin([selected_drop_location])
else: # If "All" is selected, include NaNs
    mask = mask & (df['Drop Location'].isna() | df['Drop Location'].notna())

# Booking Value filter
if 'Booking Value' in df.columns:
    mask_bv = df['Booking Value'].between(bv_range[0], bv_range[1], inclusive='both')
    # If the full booking value range is selected, also include rows where 'Booking Value' is NaN
    if bv_range[0] == float(np.nan_to_num(min_booking_value)) and bv_range[1] == float(np.nan_to_num(max_booking_value)):
        mask = mask & (mask_bv | df['Booking Value'].isna())
    else:
        mask = mask & mask_bv

df_f = df[mask].copy()

# ---------------------------
# Top KPIs
# ---------------------------
st.title("🚗 Uber Ride Analytics — 2024")
st.markdown("Interactive Streamlit dashboard: Overview, cancellations, behavior, ratings and hotspots.")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
total_bookings = len(df_f) # Corrected to use filtered df size

completed = df_f[df_f['Booking Status'].str.contains('Completed', na=False)] if 'Booking Status' in df_f.columns else df_f[df_f['Booking Status'] == 'Completed'] if 'Booking Status' in df_f.columns else df_f
completed_count = len(df_f[df_f['Booking Status'].str.contains('Completed', na=False)]) if 'Booking Status' in df_f.columns else np.nan
cancelled_count = len(df_f[df_f['Booking Status'].str.contains('Cancel', na=False)]) if 'Booking Status' in df_f.columns else np.nan
# fallback: use Cancelled Rides by Customer/Driver columns to infer cancellations
if np.isnan(cancelled_count) and 'Cancelled Rides by Customer' in df_f.columns:
    cancelled_count = int(df_f['Cancelled Rides by Customer'].sum(skipna=True) + df_f['Cancelled Rides by Driver'].sum(skipna=True))
if np.isnan(completed_count) and 'Booking Status' in df_f.columns:
    completed_count = total_bookings - cancelled_count

avg_booking_value = df_f['Booking Value'].mean() if 'Booking Value' in df_f.columns else np.nan
avg_customer_rating = df_f['Customer Rating'].mean() if 'Customer Rating' in df_f.columns else np.nan

col1.metric("Total bookings", f"{total_bookings:,}")
col2.metric("Completed", f"{int(completed_count):,}" if not np.isnan(completed_count) else "N/A",
            delta=f"{(completed_count/total_bookings*100):.1f}%" if (not np.isnan(completed_count) and total_bookings>0) else "")
col3.metric("Cancelled", f"{int(cancelled_count):,}" if not np.isnan(cancelled_count) else "N/A",
            delta=f"{(cancelled_count/total_bookings*100):.1f}%" if (not np.isnan(cancelled_count) and total_bookings>0) else "")
col4.metric("Avg. Booking Value", f"₹{avg_booking_value:,.2f}" if not np.isnan(avg_booking_value) else "N/A")

st.markdown("<hr/>", unsafe_allow_html=True)

# Booking & Cancellation Overview
with st.container():
    st.header("📊 Booking & Cancellation Patterns")
    c1, c2 = st.columns([2, 1])

    with c1:
        # time of day demand: group into bins
        df_f['time_of_day'] = pd.cut(df_f['hour'].fillna(0),
                                     bins=[-0.1,5,11,16,20,23],
                                     labels=['Late Night','Morning','Afternoon','Evening','Night'],
                                     ordered=True)
        tod_counts = df_f.groupby('time_of_day').size().reset_index(name='count')
        fig_tod = px.bar(tod_counts, x='time_of_day', y='count', title="Demand by Time of Day", text='count',
                         color_discrete_sequence=current_palette, template=template)
        fig_tod.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_tod, use_container_width=True)

        # weekdays vs weekends
        weekday_counts = df_f.groupby('weekday').size().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index(name='count')
        fig_week = px.bar(weekday_counts, x='weekday', y='count', title="Bookings by Weekday", text='count',
                          color_discrete_sequence=current_palette, template=template)
        fig_week.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_week, use_container_width=True)

    with c2:
        # Monthly trend
        monthly = df_f.groupby(['month','month_name']).size().reset_index(name='count').sort_values('month')
        fig_month = px.line(monthly, x='month_name', y='count', markers=True, title="Monthly Booking Trend",
                            color_discrete_sequence=current_palette, template=template)
        st.plotly_chart(fig_month, use_container_width=True)

        # Cancellation Pie Chart
        if 'Cancelled Rides by Customer' in df_f.columns or 'Cancelled Rides by Driver' in df_f.columns:
            cust_cancel = df_f['Cancelled Rides by Customer'].sum(skipna=True) if 'Cancelled Rides by Customer' in df_f.columns else 0
            drv_cancel = df_f['Cancelled Rides by Driver'].sum(skipna=True) if 'Cancelled Rides by Driver' in df_f.columns else 0
            cc_df = pd.DataFrame({'type':['Customer', 'Driver'], 'count':[cust_cancel, drv_cancel]})
            fig_cc = px.pie(cc_df, names='type', values='count', title='Customer vs Driver Cancellations', hole=0.4,
                            color_discrete_sequence=current_palette, template=template)
            st.plotly_chart(fig_cc, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Visualizations
with st.container():
    st.header("📈 Deeper Insights")
    tab1, tab2, tab3 = st.tabs(["Revenue & Vehicle Analysis", "Time-based Heatmap", "Customer Behavior"])

    with tab1:
        # Revenue per vehicle type
        if 'Booking Value' in df_f.columns and 'Vehicle Type' in df_f.columns:
            rev = df_f.groupby('Vehicle Type')['Booking Value'].sum().reset_index().sort_values('Booking Value', ascending=False)
            fig_rev = px.bar(rev, x='Vehicle Type', y='Booking Value', title='Revenue per Vehicle Type', text='Booking Value',
                             color_discrete_sequence=current_palette, template=template)
            fig_rev.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_rev, use_container_width=True)

        # Boxplots to compare trip fares across vehicle types
        if 'Booking Value' in df_f.columns and 'Vehicle Type' in df_f.columns:
            fig_box = px.box(df_f, x='Vehicle Type', y='Booking Value', title='Booking Value Distribution by Vehicle Type', points="outliers",
                             color='Vehicle Type', color_discrete_sequence=current_palette, template=template)
            st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        # Heatmap: cancellation rate vs time of day (hour) x weekday
        if 'Booking Status' in df_f.columns:
            pivot = df_f.assign(is_cancel = df_f['Booking Status'].str.contains('Cancel', na=False)).groupby(['hour','weekday']).agg(is_cancel_mean=('is_cancel','mean')).reset_index()
            weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pivot['weekday'] = pd.Categorical(pivot['weekday'], categories=weekday_order, ordered=True)
            pivot = pivot.sort_values(['hour','weekday'])
            heat = pivot.pivot(index='hour', columns='weekday', values='is_cancel_mean').fillna(0)
            fig_heat = px.imshow(heat.T, labels=dict(x="Hour of day", y="Weekday", color="Cancellation Rate"),
                                x=heat.index, y=heat.columns, title="Cancellation Rate Heatmap (Hour vs Weekday)",
                                color_continuous_scale=px.colors.sequential.Plasma,
                                template=template)
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        # Repeat vs one-time users
        if 'Customer ID' in df_f.columns:
            cust_counts = df_f.groupby('Customer ID').size().reset_index(name='bookings')
            one_time = (cust_counts['bookings'] == 1).sum()
            repeat = (cust_counts['bookings'] > 1).sum()
            cust_df = pd.DataFrame({'segment': ['One-time','Repeat'], 'count':[one_time, repeat]})
            fig_custseg = px.pie(cust_df, names='segment', values='count', title='Customer Segments: One-time vs Repeat', hole=0.35,
                                 color_discrete_sequence=current_palette, template=template)
            st.plotly_chart(fig_custseg, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Map Visualizations
with st.container():
    st.header("🗺️ Pickup & Dropoff Hotspots (Delhi NCR)")
    map_tab1, map_tab2 = st.tabs(["Pickup Hotspots", "Dropoff Hotspots"])

    # data for mapbox
    pickup_map_data = df_f.groupby(['Pickup Location', 'Pickup_Latitude', 'Pickup_Longitude']).size().reset_index(name='count').dropna()
    drop_map_data = df_f.groupby(['Drop Location', 'Drop_Latitude', 'Drop_Longitude']).size().reset_index(name='count').dropna()

    with map_tab1:
        if not pickup_map_data.empty:
            fig_pickup_map = px.scatter_mapbox(pickup_map_data,
                                               lat="Pickup_Latitude",
                                               lon="Pickup_Longitude",
                                               size="count",
                                               color="count",
                                               color_continuous_scale=px.colors.sequential.Plasma,
                                               zoom=10,
                                               hover_name="Pickup Location",
                                               hover_data={'count':True, 'Pickup_Latitude':False, 'Pickup_Longitude':False},
                                               title="Pickup Hotspots",
                                               mapbox_style="carto-darkmatter")
            fig_pickup_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig_pickup_map, use_container_width=True)
            st.write("Top 10 Pickup Locations:")
            st.dataframe(pickup_map_data.sort_values('count', ascending=False).head(10).set_index('Pickup Location'))
        else:
            st.warning("No pickup coordinates available for the selected filters.")

    with map_tab2:
        if not drop_map_data.empty:
            fig_drop_map = px.scatter_mapbox(drop_map_data,
                                              lat="Drop_Latitude",
                                              lon="Drop_Longitude",
                                              size="count",
                                              color="count",
                                              color_continuous_scale=px.colors.sequential.Plasma,
                                              zoom=10,
                                              hover_name="Drop Location",
                                              hover_data={'count':True, 'Drop_Latitude':False, 'Drop_Longitude':False},
                                              title="Dropoff Hotspots",
                                              mapbox_style="carto-darkmatter")
            fig_drop_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig_drop_map, use_container_width=True)
            st.write("Top 10 Dropoff Locations:")
            st.dataframe(drop_map_data.sort_values('count', ascending=False).head(10).set_index('Drop Location'))
        else:
            st.warning("No dropoff coordinates available for the selected filters.")

st.markdown("<hr/>", unsafe_allow_html=True)

# Data Table & Export
with st.container():
    st.header("💾 Data Explorer")
    st.dataframe(df_f.head(200))
    
    @st.cache_data
    def convert_df_to_csv(df_local):
        return df_local.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_f)
    st.download_button(label="Download filtered data as CSV", data=csv, file_name='uber_filtered.csv', mime='text/csv')

st.markdown("---")
st.caption("Dashboard generated from: Uber_DT_Sorted.csv, pickup_location_coords_delhi.csv, drop_location_coords_delhi.csv")

# -----------------------------------------------------------------------------
# Notes & Assumptions
# -----------------------------------------------------------------------------
with st.expander("Notes & Assumptions (click to open)"):
    st.markdown("""
    - This dashboard expects these columns (case-sensitive names as provided):
      `Date, Time, Booking ID, Booking Status, Customer ID, Vehicle Type, Pickup Location, Drop Location,
      Avg VTAT, Avg CTAT, Cancelled Rides by Customer, Cancelled Rides by Driver, Booking Value,
      Ride Distance, Driver Ratings, Customer Rating, Payment Method, datetime, Day`.
    - If `datetime` is missing, it tries to parse Date+Time.
    - `Driver ID` is not present by default in your feature list; driver-level completion rates require a driver identifier.
    - Mapping uses the pickup/drop coordinates CSVs you supplied. Ensure location names match exactly (leading/trailing spaces removed).
    - For density/heatmap mapbox visuals you would normally add a Mapbox token. This dashboard uses `st.map` and `plotly`'s `scatter_geo` to avoid requiring tokens.
    - If dataset is large and interactive performance is slow, use the sidebar filters to reduce the date range or sample size.
    - **Key Findings from Analysis (2024 Uber Ride Data):**
        - **Overall Booking & Cancellation:**
            - Total Bookings: 148,770 rides.
            - Success Rate: 65.96% (93,000 completed rides).
            - Cancellation Rate: 25% (37,430 cancelled bookings).
            - Customer Cancellations: 19.15% (27,000 rides).
            - Driver Cancellations: 7.45% (10,500 rides).
            - Null values in cancellation/incomplete ride columns indicate no cancellation/incomplete ride occurred.
            - Overall bookings and cancelled rides per day show similar fluctuation patterns.
        - **Time-based Patterns:**
            - Highest demand is in the Evening (6PM-12AM) and Afternoon (12PM-6PM).
            - Lowest demand is during Late Night (12AM-6AM).
            - This demand pattern holds across all booking statuses.
            - Monthly and weekly booking trends show some variability but no strong seasonal peaks or troughs.
            - Cancellation rates are relatively consistent across all hours of the day (around 24-26%).
        - **Cancellations (Customer vs. Driver):**
            - Driver cancellations are significantly higher than customer cancellations across all weekdays and vehicle types.
            - Auto, Go Mini, and Go Sedan vehicle types experience the highest total cancellations.
            - Uber XL has the lowest cancellations.
        - **Customer & Driver Behavior:**
            - A large majority of customers are one-time users (147,582) compared to repeat users (1,206).
            - Occasional (one-time) customers have a slightly higher success rate (62.01%) than Frequent (repeat) customers (61.37%).
            - Frequent customers show a slightly higher cancellation rate (25.64%) compared to occasional customers (24.99%).
        - **Satisfaction & Performance:**
            - Customer ratings for completed rides are generally high (peaking around 4.5-5.0).
            - Past cancellations do not significantly impact subsequent customer or driver ratings for completed rides.
            - Customer and driver ratings are consistently high across all vehicle types, with no drastic differences.
            - Price (Booking Value) and Ride Distance have minimal direct correlation with customer ratings.
            - Customer ratings are consistently high across all time segments (Morning, Afternoon, Evening, Late Night).
        - **Geographical Hotspots:**
            - Map visualizations identify specific geographical hotspots in Delhi NCR for both pickups and drop-offs, with higher concentrations in certain areas.
        - **Observations on Data Realism (Potential Artificiality):**
            - Several columns (`Cancelled Rides by Customer`, `Cancelled Rides by Driver`, `Incomplete Rides`) show descriptive statistics (mean, min, max, quartiles) of `1.0` and standard deviation of `0.0` when present. This suggests these columns might function as binary flags (1 for occurrence, NaN otherwise) rather than actual counts, or indicate a simplified data generation process.
            - The `Booking ID` and `Customer ID` columns sometimes show a `freq` of `3` for their `top` unique values, which is an unusual pattern for real-world identifiers and might be an artifact of how the dataset was created.
            - Customer and Driver Ratings, while generally high, exhibit relatively low variance in box plots and descriptive statistics, which could suggest a less organic distribution than typically found in real user feedback.
            - The ride `fare/Booking Value` is incorrect/random, as `Rs.57` is pretty less even for an `Auto` when its covering `49.98 KM` (`Booking ID: CNR8958115`) hence, the `Booking Value` might not always align with realistic Uber pricing for Delhi NCR, suggesting potential artificiality in these figures. For example, a 49km ride might have a non-sensical fare.
            - Overall, some data patterns suggest the dataset might be synthetically generated or heavily processed, rather than reflecting raw, organic Uber operational data.
    """)