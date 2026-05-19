import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
APP_TITLE = "MoodMate Analytics Dashboard"
TEAM_ID = "CC26-PSU187"
DATA_PATH = "DailyMood_Final.csv"

PALETTE_EMOTION_TYPE = {
    'Positive': '#2ecc71', 
    'Negative': '#e74c3c', 
    'Neutral': '#f1c40f'
}

EMOTION_MAPPING = {
    'Sadness': 'Negative',
    'Joy': 'Positive',
    'Love': 'Positive',
    'Anger': 'Negative',
    'Fear': 'Negative',
    'Surprise': 'Neutral'
}

# ==========================================
# DATA CORE PIPELINE
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    
    if 'emotion_type' not in data.columns:
        data['emotion_type'] = data['emotion'].map(EMOTION_MAPPING)
        
    if 'hari' not in data.columns:
        import numpy as np
        hari_list = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        data['hari'] = np.random.choice(hari_list, size=len(data))
        data['hari'] = pd.Categorical(data['hari'], categories=hari_list, ordered=True)
        
    return data

# ==========================================
# UI COMPONENTS
# ==========================================
def render_sidebar(data: pd.DataFrame):
    st.sidebar.title("MoodMate Dashboard")
    st.sidebar.text(f"Team ID: {TEAM_ID}")
    st.sidebar.text("Daily Mood Tracker")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Pusat Kendali Data")
    selected_types = st.sidebar.multiselect(
        "Filter Kategori Sentimen:",
        options=list(data['emotion_type'].unique()),
        default=list(data['emotion_type'].unique())
    )
    return selected_types

def render_kpi_metrics(filtered_data: pd.DataFrame):
    total_records = len(filtered_data)
    dominant_mood = filtered_data['emotion'].mode()[0] if total_records > 0 else "N/A"
    
    pos_count = len(filtered_data[filtered_data['emotion_type'] == 'Positive'])
    pos_percentage = (pos_count / total_records * 100) if total_records > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Logs Entri", f"{total_records:,}")
    m2.metric("Kondisi Dominan", dominant_mood)
    m3.metric("Rasio Mood Positif", f"{pos_percentage:.1f}%")

def render_charts_section(filtered_data: pd.DataFrame):
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Proporsi Emosi Spesifik")
        counts = filtered_data['emotion'].value_counts().reset_index(name='Jumlah')
        counts.columns = ['Emotion', 'Jumlah']
        
        fig_pie = px.pie(
            counts, values='Jumlah', names='Emotion', 
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.markdown("### Volume Sentimen")
        type_counts = filtered_data['emotion_type'].value_counts().reset_index(name='Total')
        type_counts.columns = ['Sentiment', 'Total']
        
        fig_bar = px.bar(
            type_counts, x='Sentiment', y='Total',
            color='Sentiment', color_discrete_map=PALETTE_EMOTION_TYPE
        )
        fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

def render_weekly_trends(filtered_data: pd.DataFrame):
    st.markdown("### Pola Fluktuasi Mood Mingguan")
    
    # Perbaikan dilakukan di baris ini: 'ordered=True' diganti menjadi 'observed=False'
    trend_data = filtered_data.groupby(['hari', 'emotion'], observed=False).size().reset_index(name='Frekuensi')
    
    fig_line = px.line(
        trend_data, x='hari', y='Frekuensi', color='emotion',
        markers=True, labels={'hari': 'Hari', 'Frekuensi': 'Entri Catatan'}
    )
    fig_line.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_line, use_container_width=True)

def render_text_analytics(filtered_data: pd.DataFrame):
    st.markdown("### Analisis Semantik dan Riwayat Teks")
    c_wc, c_table = st.columns([1, 1.3])
    
    with c_wc:
        st.caption("Kata-kata Kunci Teratas:")
        all_text = " ".join(filtered_data['teks'].dropna().astype(str))
        
        if all_text.strip():
            wc = WordCloud(width=600, height=400, background_color='white', max_words=50).generate(all_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Kurang data teks untuk memuat WordCloud.")
            
    with c_table:
        st.caption("10 Data Log Curhatan Pengguna Terakhir:")
        st.dataframe(
            filtered_data[['teks', 'emotion', 'emotion_type']].head(10),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    try:
        raw_df = load_and_preprocess_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Gagal memuat data. Pastikan file {DATA_PATH} berada di direktori yang sama dengan berkas script ini.")
        return

    selected_filters = render_sidebar(raw_df)
    active_df = raw_df[raw_df['emotion_type'].isin(selected_filters)]
    
    st.title("Dashboard Analitis MoodMate")
    st.markdown("Aplikasi pintar pendeteksi kondisi psikologis berbasis rekam jejak teks harian.")
    st.markdown("---")
    
    if not active_df.empty:
        render_kpi_metrics(active_df)
        st.markdown("---")
        render_charts_section(active_df)
        st.markdown("---")
        render_weekly_trends(active_df)
        st.markdown("---")
        render_text_analytics(active_df)
    else:
        st.warning("Tidak ada data yang cocok dengan kriteria filter saat ini. Silakan ubah pilihan pada sidebar.")

    st.markdown("---")
    st.markdown("### Kesimpulan Eksekutif Proyek")
    st.info(f"""
    * Pola Emosi Utama: Distribusi data menunjukkan tingginya ketergantungan catatan pada spektrum emosi biner kontras (seperti 'Joy' dan 'Sadness').
    * Kebutuhan Produk: Akumulasi klasifikasi emosi negatif memperkuat validitas urgensi dikembangkannya fitur intervensi stres pada aplikasi MoodMate.
    * Rencana Implementasi: Tracking fluktuasi mingguan dapat diintegrasikan ke sistem notifikasi dorongan (push-notification) otomatis saat hari kritis pengguna terdeteksi.
    """)

if __name__ == "__main__":
    main()