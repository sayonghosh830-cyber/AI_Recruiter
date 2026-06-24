import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import hashlib
import plotly.express as px
from pathlib import Path

# Absolute directory path setup to ensure flawless execution under virtualenvs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set page config with deep dark recruiter theme layout
st.set_page_config(
    page_title="Hiring Priority & Hybrid Ranking Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Recruiter Dark Theme via Custom CSS injection
st.markdown("""
<style>
    /* Custom dark theme container overrides */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-header {
        font-size: 0.9rem;
        color: #8b949e;
        margin-bottom: 4px;
        text-transform: uppercase;
        tracking: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .candidate-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .candidate-card:hover {
        border-color: #58a6ff;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-exceptional { background-color: rgba(46, 160, 67, 0.15); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
    .badge-recommended { background-color: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.4); }
    .badge-caution { background-color: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.4); }
    .badge-notaligned { background-color: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
</style>
""", unsafe_allow_html=True)

st.title("💼 Recruiter Priority & Hybrid Candidate Ranking Board")
st.caption("Dynamic talent prioritization based on target skills overlap, semantic similarity, career stability, and engagement signals")

# Side bar controls
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    # Rule 2: Dynamic File Parsing with explicit type filter
    uploaded_file = st.file_uploader("Upload Candidates JSON", type=["json"])
    
    st.subheader("💡 Job Parameter Reference")
    config_path = Path(os.path.join(CURRENT_DIR, "config.json"))
    job_title = "Backend Software Engineer"
    required_skills = []
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            job_title = cfg['target_job'].get('job_title', job_title)
            required_skills = cfg['target_job'].get('required_skills', [])
            st.markdown(f"**Target Role:** `{job_title}`")
            st.markdown("**Required Toolkit:**")
            skills_md = " ".join([f"`{s}`" for s in required_skills])
            st.markdown(skills_md)
        except Exception as e:
            st.sidebar.error(f"Error reading job parameters: {e}")

# Ensure dynamic data directory exists
os.makedirs(os.path.join(CURRENT_DIR, "data"), exist_ok=True)
active_candidates_path = os.path.join(CURRENT_DIR, "data", "active_candidates.json")

# Rule 1: No default lists, empty or informational state when no file uploaded
if uploaded_file is None:
    st.info("👋 Welcome! Please upload a candidate pool JSON file using the sidebar to begin candidate priority ranking.")
    
    st.markdown("""
    ### 🚀 Getting Started with the Prioritization Pipeline
    
    This platform prioritizes talent pools based on an end-to-end recruitment prioritization and matching model. 
    To evaluate a candidate pool:
    
    1. **Prepare your Candidates JSON file**: Make sure it matches the expected schema containing `candidate_id`, `profile`, `skills`, and `redrob_signals`.
    2. **Upload the file**: Use the file uploader widget in the left sidebar.
    3. **Analyze Results**: The pipeline will automatically parse, score, rank, and generate an interactive dashboard of candidate insights.
    4. **Export Submission**: Export a valid CSV file of the calculated rankings directly to your system.
    """)
    st.stop()

# Rule 2: Dynamically parse the candidate JSON file
try:
    file_bytes = uploaded_file.getvalue()
    candidates_data = json.loads(file_bytes.decode("utf-8"))
except Exception as e:
    st.error(f"⚠️ Failed to parse uploaded JSON file: {e}")
    st.stop()

# Rule 4: Handle edge case (zero candidates)
if not isinstance(candidates_data, list) or len(candidates_data) == 0:
    st.warning("⚠️ The uploaded file contains 0 participants. Please upload a valid candidate pool.")
    st.stop()

# Rule 3: Integrate with scoring engine (write parsed JSON and run pipeline using hash cache)
file_hash = hashlib.md5(file_bytes).hexdigest()

if "last_processed_hash" not in st.session_state or st.session_state["last_processed_hash"] != file_hash:
    with st.spinner("Processing matching layer and ranking hybrid weights..."):
        try:
            # Save parsed json to active path for scoring engine
            with open(active_candidates_path, "w", encoding="utf-8") as f:
                json.dump(candidates_data, f, indent=2)
            
            # Trigger our end-to-end pipeline execution
            python_executable = sys.executable if sys.executable else "python3"
            pipeline_path = os.path.join(CURRENT_DIR, "main_pipeline.py")
            result = subprocess.run(
                [python_executable, pipeline_path],
                cwd=CURRENT_DIR,
                capture_output=True,
                text=True,
                check=True
            )
            st.session_state["last_processed_hash"] = file_hash
            st.session_state["pipeline_success"] = True
            st.toast("Pipeline executed successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline execution failed: {e}")
            if e.stderr:
                st.code(e.stderr)
            st.session_state["pipeline_success"] = False
            st.stop()
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")
            st.session_state["pipeline_success"] = False
            st.stop()

# Load calculated rankings from CSV output
leaderboard_csv = os.path.join(CURRENT_DIR, "data", "ranked_leaderboard.csv")
if os.path.exists(leaderboard_csv):
    try:
        df = pd.read_csv(leaderboard_csv)
    except Exception as e:
        st.error(f"Error reading compiled leaderboard: {e}")
        st.stop()
else:
    st.error("Leaderboard file was not generated by the pipeline.")
    st.stop()

# Rule 5: Render Dynamic Leaderboard & Download
if not df.empty:
    # 1. KPI Cards Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_profiles = len(df)
    top_score = df["final_match_score"].max() if "final_match_score" in df.columns else 0.0
    avg_score = df["final_match_score"].mean() if "final_match_score" in df.columns else 0.0
    rec_count = len(df[df["hiring_confidence"].isin(["EXCEPTIONAL", "RECOMMENDED"])]) if "hiring_confidence" in df.columns else 0
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">Total Profiles Analyzed</div>
            <div class="kpi-value">{total_profiles}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">Top Scoring Candidate</div>
            <div class="kpi-value">{top_score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">Average Fit Score</div>
            <div class="kpi-value">{avg_score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">Recommended Alignments</div>
            <div class="kpi-value">{rec_count}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Plotly Visualizations Section
    st.subheader("📊 Recruiter Insights & Performance Distribution")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Core Fit Score & Recommendation Distribution**")
        if "hiring_confidence" in df.columns:
            conf_counts = df["hiring_confidence"].value_counts().reset_index()
            conf_counts.columns = ["Confidence Tier", "Count"]
            color_map = {
                "EXCEPTIONAL": "#3fb950",
                "RECOMMENDED": "#58a6ff",
                "CONSIDER_WITH_CAUTION": "#d29922",
                "NOT_ALIGNED": "#f85149"
            }
            fig_pie = px.pie(
                conf_counts, 
                names="Confidence Tier", 
                values="Count", 
                hole=0.45,
                color="Confidence Tier",
                color_discrete_map=color_map,
                category_orders={"Confidence Tier": ["EXCEPTIONAL", "RECOMMENDED", "CONSIDER_WITH_CAUTION", "NOT_ALIGNED"]}
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#c9d1d9",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col_chart2:
        st.markdown("**Top 10 Candidate Alignment Weights**")
        if "final_match_score" in df.columns:
            df_top_10 = df.head(10).copy()
            df_top_10 = df_top_10.iloc[::-1] # reverse so top candidates show first in horizontal bar
            color_discrete_map = {
                "EXCEPTIONAL": "#3fb950",
                "RECOMMENDED": "#58a6ff",
                "CONSIDER_WITH_CAUTION": "#d29922",
                "NOT_ALIGNED": "#f85149"
            }
            fig_bar = px.bar(
                df_top_10,
                x="final_match_score",
                y="name",
                orientation="h",
                color="hiring_confidence",
                color_discrete_map=color_discrete_map,
                labels={"final_match_score": "Composite Core Score", "name": "Candidate"},
                hover_data=["leaderboard_rank", "suggested_action"]
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#c9d1d9",
                xaxis=dict(
                    showgrid=True, 
                    gridcolor="#30363d", 
                    range=[0, 1.0],
                    title=dict(
                        text="Composite Core Score",
                        standoff=15
                    )
                ),
                yaxis=dict(showgrid=False, automargin=True),
                margin=dict(l=160, r=20, t=20, b=80),
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.35, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Dynamic Leaderboard Dataframe (Rule 5 requirement)
    st.subheader("🏆 Live Calculated Leaderboard")
    
    # Map, format and extract specific columns as requested
    df_display = pd.DataFrame()
    df_display["candidate_id"] = df["candidate_id"]
    df_display["anonymized_name"] = df["name"]
    df_display["fit_score"] = df["final_match_score"].round(4)
    df_display["explainable_reason"] = df["one_liner_reasoning"]
    
    st.dataframe(df_display, use_container_width=True)

    # 4. Detailed Candidate Evaluation Explorers
    st.subheader("🔍 Selected Candidate Summaries & Evaluation Explanations")
    df_top = df.head(25)
    
    for idx, row in df_top.iterrows():
        cand_id = row.get("candidate_id", "Unknown")
        name = row.get("name", "Anonymized Profile")
        score = row.get("final_match_score", 0.0)
        rank = row.get("leaderboard_rank", idx + 1)
        tier = row.get("hiring_confidence", "NOT_ALIGNED")
        reasoning = row.get("one_liner_reasoning", "No explicit reason loaded.")
        action = row.get("suggested_action", "No dynamic hiring action recommendation provided.")
        matched = row.get("matched_skills", "None detected")
        missing = row.get("missing_skills", "None")
        
        badge_cls = "badge-notaligned"
        if tier == "EXCEPTIONAL":
            badge_cls = "badge-exceptional"
        elif tier == "RECOMMENDED":
            badge_cls = "badge-recommended"
        elif tier == "CONSIDER_WITH_CAUTION":
            badge_cls = "badge-caution"
            
        header_text = f"Rank #{rank} - {name} ({cand_id}) | Score: {score:.4f}"
        
        with st.expander(header_text):
            st.markdown(f"""
            <div style="padding-top: 5px; padding-bottom: 5px;">
                <span class="badge {badge_cls}">{tier}</span>
                <span style="color:#8b949e; margin-left: 20px;">**Suggested Action:** {action}</span>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Matched Tech Stacks / Skills:**")
                st.write(matched if str(matched) != "nan" and matched else "No verified core matches.")
            with c2:
                st.markdown("**⚠️ Key Tool Gaps (Missing Targets):**")
                st.write(missing if str(missing) != "nan" and missing else "No identified gaps.")
                
            st.markdown("**🎯 Evaluation Rationale:**")
            st.info(reasoning)
            
            st.markdown("**🔬 Metric Diagnostic Profile:**")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            
            stab = row.get("stability_score", row.get("tenure_stability_score", 0.0))
            sc1.metric("Tenure Stability", f"{float(stab):.2f}" if pd.notna(stab) else "0.00")
            
            acad = row.get("academic_prestige", row.get("academic_prestige_score", 0.0))
            sc2.metric("Academic Prestige", f"{float(acad):.2f}" if pd.notna(acad) else "0.00")
            
            resp = row.get("platform_responsiveness", row.get("responsiveness_score", 0.0))
            sc3.metric("Platform Responsiveness", f"{float(resp):.2f}" if pd.notna(resp) else "0.00")
            
            attend = row.get("attendance_reliability", row.get("reliability_score", 0.0))
            sc4.metric("Attendance Reliability", f"{float(attend):.2f}" if pd.notna(attend) else "0.00")
            
            notice = row.get("notice_period_days", row.get("notice_period", "Immediate"))
            sc5.metric("Notice Period", f"{int(float(notice))} Days" if pd.notna(notice) and str(notice).strip() != '' else "Immediate")

    # 5. Export calculated results as submission (Rule 5 requirement)
    st.subheader("💾 Export Dynamic Analysis & Submissions")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Rich Analysis (CSV)",
            data=csv_data,
            file_name="ranked_leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_d2:
        # Generate official submission format: candidate_id, rank, score, reasoning
        submission_df = pd.DataFrame()
        submission_df["candidate_id"] = df["candidate_id"]
        submission_df["rank"] = df["leaderboard_rank"]
        submission_df["score"] = df["final_match_score"].round(4)
        submission_df["reasoning"] = df["one_liner_reasoning"]
        
        submission_csv_data = submission_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🎯 Download Platform Submission File (submission.csv)",
            data=submission_csv_data,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True
        )
