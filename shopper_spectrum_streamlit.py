# Import Libraries 
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# ── Load saved models ─────────────────────────────────────────
@st.cache_resource
def load_models():
    import os
    from sklearn.metrics.pairwise import cosine_similarity
    with open('kmeans_model.pkl', 'rb') as f:
        km = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Rebuild similarity matrix if pkl not present
    if not os.path.exists('item_similarity.pkl'):
        df = pd.read_csv('online_retail.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df = df.dropna(subset=['CustomerID'])
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        df = df.drop_duplicates()
        pivot = df.groupby(['CustomerID', 'Description'])['Quantity'].sum().unstack().fillna(0)
        item_sim = cosine_similarity(pivot.T)
        item_sim_df = pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)
    else:
        item_sim_df = pd.read_pickle('item_similarity.pkl')
    return km, scaler, item_sim_df
km, scaler, item_sim_df = load_models()  # ← keep this line, don't remove it

@st.cache_data
def load_rfm():
    return pd.read_csv('rfm_data.csv')

rfm = load_rfm()

# Cluster label mapping (based on your cluster means output)
cluster_label_map = {
    0: '🟡 Regular',
    1: '🔴 At-Risk',
    2: '🟢 High-Value',
    3: '🔵 Occasional'
}

# ── Sidebar Navigation ────────────────────────────────────────
#pages = ["🏠 Home", "👤 Customer Segmentation", "🎯 Product Recommendation"]
pages = ["🏠 Home", "📊 RFM Analysis", "👤 Customer Segmentation", "🎯 Product Recommendation"]

page = st.sidebar.radio("Navigate to", pages)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation & Product Recommendation System")
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 👤 Customer Segmentation\nEnter Recency, Frequency, and Monetary values to predict which customer segment a customer belongs to.")
        st.markdown("👈 **Use the sidebar to navigate to Segmentation**")

    with col2:
        st.success("### 🎯 Product Recommendation\nEnter a product name to get 5 similar product recommendations based on purchase history.")
        st.markdown("👈 **Use the sidebar to navigate to Recommendation**")

    st.markdown("---")
    st.markdown("#### Customer Segments at a Glance")
    seg_data = {
        "Segment": ["🟢 High-Value", "🟡 Regular", "🔵 Occasional", "🔴 At-Risk"],
        "Recency": ["Very Recent", "Recent", "Recent", "Long Ago"],
        "Frequency": ["Very High", "Low", "Medium", "Very Low"],
        "Monetary": ["Very High", "Low", "Medium", "Low"],
        "Description": [
            "Frequent, recent, big spenders",
            "Steady but not premium buyers",
            "Moderate purchasers",
            "Haven't purchased in a long time"
        ]
    }
    st.table(pd.DataFrame(seg_data))

# Add this line below the table on Home page
    st.markdown("👈 **Navigate to RFM Analysis in the sidebar to explore detailed charts**")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — RFM ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📊 RFM Analysis":
    st.title("📊 RFM Analysis")
    st.markdown("Visual breakdown of customer Recency, Frequency and Monetary scores.")
    st.markdown("---")

    import matplotlib.pyplot as plt

    # ── Chart 1: Recency Distribution ─────────────────────────
    st.subheader("1️⃣ Recency Distribution")
    st.markdown("How recently did customers make a purchase?")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(rfm['Recency'], bins=50, color='steelblue', edgecolor='white')
    ax1.set_xlabel("Recency (days)")
    ax1.set_ylabel("Number of Customers")
    ax1.set_title("Recency Distribution")
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("---")

    # ── Chart 2: Frequency Distribution ───────────────────────
    st.subheader("2️⃣ Frequency Distribution")
    st.markdown("How many times did customers purchase?")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(rfm['Frequency'], bins=50, color='seagreen', edgecolor='white')
    ax2.set_xlabel("Frequency (number of purchases)")
    ax2.set_ylabel("Number of Customers")
    ax2.set_title("Frequency Distribution")
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")

    # ── Chart 3: Monetary Distribution ────────────────────────
    st.subheader("3️⃣ Monetary Distribution")
    st.markdown("How much did customers spend in total?")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.hist(rfm['Monetary'], bins=50, color='coral', edgecolor='white')
    ax3.set_xlabel("Monetary (total spend £)")
    ax3.set_ylabel("Number of Customers")
    ax3.set_title("Monetary Distribution")
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("---")

    # ── Chart 4: Customer Segment Distribution ─────────────────
    st.subheader("4️⃣ Customer Segment Distribution")
    st.markdown("How many customers fall in each segment?")
    segment_counts = rfm['Segment'].value_counts()
    colors = ['#2ecc71', '#f1c40f', '#3498db', '#e74c3c']
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    bars = ax4.bar(segment_counts.index, segment_counts.values,
                   color=colors[:len(segment_counts)], edgecolor='white')
    ax4.set_xlabel("Segment")
    ax4.set_ylabel("Number of Customers")
    ax4.set_title("Customer Segment Distribution")
    for bar, val in zip(bars, segment_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")

    # ── Summary Table ──────────────────────────────────────────
    st.subheader("📋 RFM Summary by Segment")
    summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(2)
    st.dataframe(summary, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════
elif page == "👤 Customer Segmentation":
    st.title("👤 Customer Segmentation")
    st.markdown("Enter the customer's RFM values to predict their segment.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("📅 Recency (days since last purchase)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=1, value=5)
    with col3:
        monetary = st.number_input("💰 Monetary (total spend £)", min_value=0.0, value=500.0, step=50.0)

    st.markdown("---")

    if st.button("🔍 Predict Segment", use_container_width=True):
        # scaled = scaler.transform([[recency, frequency, monetary]])
        input_df = pd.DataFrame([[recency, frequency, monetary]], 
                         columns=['Recency', 'Frequency', 'Monetary'])
        scaled = scaler.transform(input_df)
        cluster = km.predict(scaled)[0]
        label = cluster_label_map.get(cluster, "Unknown")

        st.markdown("### Result")
        if "High-Value" in label:
            st.success(f"This customer belongs to: **{label}**")
            st.markdown("💡 *Reward with loyalty programs and exclusive offers.*")
        elif "Regular" in label:
            st.info(f"This customer belongs to: **{label}**")
            st.markdown("💡 *Encourage upselling with bundle deals.*")
        elif "Occasional" in label:
            st.warning(f"This customer belongs to: **{label}**")
            st.markdown("💡 *Re-engage with targeted promotions.*")
        elif "At-Risk" in label:
            st.error(f"This customer belongs to: **{label}**")
            st.markdown("💡 *Send win-back campaigns immediately.*")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT RECOMMENDATION
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Product Recommendation":
    st.title("🎯 Product Recommender")
    st.markdown("Enter a product name to find 5 similar products.")
    st.markdown("---")

    product_input = st.text_input("🔎 Enter Product Name (in UPPERCASE)", 
                                   placeholder="e.g. GREEN VINTAGE SPOT BEAKER")

    if st.button("✨ Get Recommendations", use_container_width=True):
        product_input = product_input.strip().upper()

        if product_input == "":
            st.warning("Please enter a product name.")
        elif product_input not in item_sim_df.columns:
            st.error(f"Product '{product_input}' not found in the database.")
            st.markdown("**Try one of these:**")
            # Show 5 sample product names as hint
            samples = list(item_sim_df.columns[:5])
            for s in samples:
                st.write(f"• {s}")
        else:
            sim_scores = item_sim_df[product_input].sort_values(ascending=False)
            recommendations = sim_scores.iloc[1:6].index.tolist()

            st.markdown(f"### Recommended products similar to **{product_input}**:")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style='padding:10px; margin:5px 0; background-color:#f0f2f6; 
                border-radius:8px; font-size:16px'>
                    <b>{i}.</b> {rec}
                </div>
                """, unsafe_allow_html=True)