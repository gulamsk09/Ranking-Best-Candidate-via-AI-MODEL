import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

# To Load Engine File
import engine  

st.set_page_config(page_title="Candidate Discovery System", page_icon="🚀", layout="wide")

@st.cache_resource
def load_shared_encoder() -> SentenceTransformer:
    """Save Data and Load Modell"""
    return SentenceTransformer("all-MiniLM-L6-v2")

encoder_model = load_shared_encoder()

# UI HEADING on Screen while running
st.title("🚀 Intelligent Candidate Discovery System")
st.markdown(" Skill Synergy based smart hiring platform with AI Explainability.")
st.markdown("Developer: Mohd Gulam Moinuddin Ramzan Sheikh.")
st.write("---")

# Divide screen Two Paets (Left & Right)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    job_description = st.text_area(
        "Enter Job Description:",
        placeholder="Example: We need a Python developer with experience in AI and Data Analysis...",
        height=180
    )

with col2:
    candidate_csv = st.file_uploader(
        "Upload Candidates CSV (with 'candidate_id' and 'reasoning' columns)", 
        type="csv"
    )

# Completion of Data Loading
if candidate_csv and job_description.strip():
    try:
        candidate_data = pd.read_csv(candidate_csv)
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        candidate_data = None

    if candidate_data is not None:
        # Checking the CSV File
        if "candidate_id" not in candidate_data.columns or "reasoning" not in candidate_data.columns:
            st.error("❌ Schema mismatch: Input CSV must contain 'candidate_id' and 'reasoning' columns.")
        else:
            with st.spinner("Analyzing candidate text matching vectors..."):
                # Engine file ko bola ki calculation shuru karo
                ranked_candidates = engine.process_and_rank_candidates(candidate_data, job_description, encoder_model)
            
            st.success("Analysis Complete!")
            st.write("---")
            st.subheader("🎯 Ranked Candidates & Breakdown")
            
            # To Show OUTPUT
            st.dataframe(
                ranked_candidates[[
                    "candidate_id",
                    "Match_Score",
                    "Match Category",
                    "Overlapping Keywords",
                    "reasoning",
                ]],
                column_config={
                    "Match_Score": st.column_config.ProgressColumn(
                        "Match Score (%)",
                        help="Semantic similarity calculated by Sentence-Transformers",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Match Category": st.column_config.TextColumn("Tier Evaluation"),
                    "Overlapping Keywords": st.column_config.TextColumn("Shared Key Skills Found"),
                    "reasoning": st.column_config.TextColumn("Candidate Profiling Notes")
                },
                use_container_width=True,
                hide_index=True,
            )
            
            # FOR Download
            export_csv = ranked_candidates.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Ranked List with Explanations",
                data=export_csv,
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )