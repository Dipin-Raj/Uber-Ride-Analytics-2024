# 🚖 Uber Ride Analytics 2024

An interactive **Streamlit dashboard** to explore **Uber ride trends, cancellations, customer behavior, ratings, and geographical hotspots** in Delhi NCR.

## ✨ Features
- 📊 **Bookings & Cancellations** — Analyze daily, weekly, and monthly booking patterns.
- ⏰ **Time Insights** — Demand by time of day, weekday trends, and cancellation heatmaps.
- 🚗 **Vehicle Analytics** — Revenue and booking value comparisons across vehicle types.
- 👥 **Customer Behavior** — One-time vs repeat users and rating trends.
- 🗺️ **Hotspot Maps** — Pickup & drop-off hotspots with interactive Mapbox visuals.
- 💾 **Data Explorer** — Filter, preview, and export rides data.

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) — Interactive dashboard framework  
- [Pandas](https://pandas.pydata.org/) — Data manipulation  
- [Plotly](https://plotly.com/python/) — Advanced visualizations  
- [NumPy](https://numpy.org/) — Numerical operations  

## 📂 Dataset
The dashboard uses:
- `Uber_DT_Sorted.csv` → Core booking dataset  
- `pickup_location_coords_delhi.csv` → Pickup coordinates  
- `drop_location_coords_delhi.csv` → Dropoff coordinates  

👉 Data may be **synthetically generated** and not reflect real Uber operational records.

## 🚀 How to Run
1. Clone the repo  
   ```bash
   git clone https://github.com/Dipin-Raj/Uber-Ride-Analytics-2024.git
   cd Uber-Ride-Analytics-2024

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run the dashboard
   ```bash
   streamlit run main.py

📊 Dashboard Preview
-🔹 KPIs: Total bookings, completed rides, cancellations, average booking value
-🔹 Trends: Time of day demand, weekday bookings, monthly booking patterns
-🔹 Insights: Revenue per vehicle, booking value distributions, repeat vs one-time customers
-🔹 Maps: Pickup and drop-off hotspots across Delhi NCR

## 📌 Notes & Assumptions
- Expects these key columns: `Date, Time, Booking ID, Booking Status, Customer ID, Vehicle Type, Pickup Location, Drop Location, Booking Value, Ride Distance, Ratings`.
- If `datetime` is missing, it parses from `Date + Time`.
- Some patterns (like fare anomalies or static cancellation flags) suggest artificial/simplified data generation.

## 📍 Author: Dipin Raj
📧 Contact: dipinr505@gmail.com

##⚡ “Turning raw ride data into actionable insights, one visualization at a time.”






