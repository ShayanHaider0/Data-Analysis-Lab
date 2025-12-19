import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Global Superstore Dashboard", layout="wide")

# ---------------------------
# Load Dataset Directly
# ---------------------------
df = pd.read_csv("data/Global_Superstore.csv", encoding="latin-1")

# Data Cleaning
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df.dropna(subset=['Sales', 'Profit'], inplace=True)

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

sub_category = st.sidebar.multiselect(
    "Select Sub-Category",
    options=df['Sub-Category'].unique(),
    default=df['Sub-Category'].unique()
)

# Apply filters
filtered_df = df[
    (df['Region'].isin(region)) &
    (df['Category'].isin(category)) &
    (df['Sub-Category'].isin(sub_category))
]

# ---------------------------
# KPI Metrics
# ---------------------------
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()

col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Total Profit", f"${total_profit:,.2f}")

st.markdown("---")

# ---------------------------
# Top 5 Customers by Sales
# ---------------------------
top_customers = (
    filtered_df.groupby('Customer Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

st.subheader("Top 5 Customers by Sales")

fig, ax = plt.subplots()
top_customers.plot(kind='bar', ax=ax, color='skyblue')
ax.set_xlabel("Customer Name")
ax.set_ylabel("Total Sales")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# ---------------------------
# Add More Charts
# ---------------------------
st.markdown("---")
st.subheader("Sales by Category")

category_sales = filtered_df.groupby('Category')['Sales'].sum()
fig2, ax2 = plt.subplots()
category_sales.plot(kind='bar', ax=ax2, color='orange')
ax2.set_ylabel("Total Sales")
st.pyplot(fig2)


