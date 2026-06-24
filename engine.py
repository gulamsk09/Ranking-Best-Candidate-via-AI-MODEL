import re
from typing import Set
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Standard line to clear keywords
STOP_WORDS: Set[str] = {
    "and", "the", "with", "for", "this", "that", "have", "from", 
    "need", "required", "requirements", "experience", "knowledge"
}

def extract_matching_keywords(jd_text: str, resume_text: str, max_keywords: int = 6) -> str:
    """
    To Prevent overlapping skill keywords, 
    
    """
    if not isinstance(jd_text, str) or not isinstance(resume_text, str):
        return ""
    
    # Using regex boundary for data clearinh
    jd_tokens = {word for word in re.findall(r'\b\w{3,}\b', jd_text.lower())}
    resume_tokens = {word for word in re.findall(r'\b\w{3,}\b', resume_text.lower())}
    
    # Now remove noise Valus
    keyword_matches = (jd_tokens & resume_tokens) - STOP_WORDS
    return ", ".join(list(keyword_matches)[:max_keywords])

def get_tier_explanation(score: float) -> str:
    """
    Categorizes the calculated cosine similarity percentage into 
    distinct performance tiers for recruitment review.
    """
    if score >= 70.0:
        return "🟢 High Match: Exceptional alignment with core role concepts, required tech stack, and background."
    elif score >= 40.0:
        return "🟡 Moderate Match: Shares several core skills but missing a few specialized requirements."
    else:
        return "🔴 Low Match: Significant gap in required skills, tech stack, or overall professional focus."

def process_and_rank_candidates(df: pd.DataFrame, jd_text: str, model: SentenceTransformer) -> pd.DataFrame:
  
    # Create local def copy to prevent chaning the original upload data
    working_df = df.copy()
    
    # Convert similarity output to percentage
    reasoning_data = working_df.get("reasoning", pd.Series([""] * len(working_df))).fillna("").astype(str)
    skills_data = working_df.get("skills", pd.Series([""] * len(working_df))).fillna("").astype(str)
    exp_data = working_df.get("experience_years", pd.Series([""] * len(working_df))).fillna("").astype(str)
    
    # Merge various features 
    working_df["combined_profile_text"] = reasoning_data + " " + skills_data + " " + exp_data
    
    # Vector generation using sentence transfirmer
    profiles_list = working_df["combined_profile_text"].tolist()
    candidate_embeddings = model.encode(profiles_list, batch_size=32, show_progress_bar=False)
    jd_embedding = model.encode(jd_text)
    
    
    similarity_tensor = util.cos_sim(jd_embedding, candidate_embeddings)
    
    # Safely extract into clear floating points array
    if hasattr(similarity_tensor, "cpu"):
        raw_scores = (similarity_tensor[0].cpu().numpy() * 100).round(2)
    else:
        raw_scores = (np.array(similarity_tensor[0]) * 100).round(2)
        
    working_df["Match_Score"] = raw_scores
    
    # Generate explainability columns based on canndidate data
    working_df["Match Category"] = working_df["Match_Score"].apply(get_tier_explanation)
    working_df["Overlapping Keywords"] = working_df["combined_profile_text"].apply(
        lambda text_block: extract_matching_keywords(jd_text, text_block)
    )
    
    # Sort dataset descending based on algorithmic match metrics
    return working_df.sort_values(by="Match_Score", ascending=False)
