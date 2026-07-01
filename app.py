import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import hashlib
import plotly.express as px
import io  # Added to handle memory buffers for Excel conversion
import openpyxl
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
    .badge-exceptional { background-color: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.4); }
    .badge-recommended { background-color: rgba(52, 152, 219, 0.15); color: #3498db; border: 1px solid rgba(52, 152, 219, 0.4); }
    .badge-caution { background-color: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid rgba(241, 196, 15, 0.4); }
    .badge-notaligned { background-color: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid rgba(231, 76, 60, 0.4); }
</style>
""", unsafe_allow_html=True)

st.title("💼 Recruiter Priority & Hybrid Candidate Ranking Board")
st.caption("Dynamic talent prioritization based on target skills overlap, semantic similarity, career stability, and engagement signals")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    # Rule 2: Dynamic File Parsing with explicit type filter
    uploaded_file = st.file_uploader("Upload Candidates JSON", type=["json"])
    
    st.subheader("💡 Customize Job Criteria")
    config_path = Path(os.path.join(CURRENT_DIR, "config.json"))
    job_title_default = "Backend Engineer"
    required_skills_default = ["Python", "SQL", "Kafka", "Spark", "Docker"]
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            job_title_default = cfg['target_job'].get('job_title', job_title_default)
            required_skills_default = cfg['target_job'].get('required_skills', required_skills_default)
        except Exception as e:
            pass

    # Dynamic UI Input Widgets using standard variables
    job_title_input = st.text_input(
        "Target Job Title", 
        value=job_title_default, 
        placeholder="Enter target job (e.g. Civil Engineer, Backend Software Engineer)..."
    )
    required_toolkit_input = st.text_input(
        "Required Toolkit", 
        value=",".join(required_skills_default) if isinstance(required_skills_default, list) else required_skills_default, 
        placeholder="Enter required tools separated by commas (e.g. AutoCAD, STAAD Pro)..."
    )
    
    # Process the string into a clean list in Python using unified required_toolkit variable
    required_toolkit = [tech.strip() for tech in required_toolkit_input.split(",") if tech.strip()] if required_toolkit_input else []

# Ensure dynamic data directory exists
os.makedirs(os.path.join(CURRENT_DIR, "data"), exist_ok=True)
active_candidates_path = os.path.join(CURRENT_DIR, "data", "active_candidates.json")

# Safety check to block execution with a warning if the job title or toolkit input is empty on page load
if not job_title_input or not required_toolkit:
    st.warning("🎯 Please enter a Target Job Title and at least one required tool in the sidebar to start ranking candidates!")
    st.stop()

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

# Rule 2: Dynamically parse the candidate JSON file with error-tolerant decoding
try:
    file_bytes = uploaded_file.getvalue()
    candidates_data = json.loads(file_bytes.decode("utf-8", errors="replace"))
except Exception as e:
    st.error(f"⚠️ Failed to parse uploaded JSON file: {e}")
    st.stop()

# Rule 4: Handle edge case (zero candidates)
if not isinstance(candidates_data, list) or len(candidates_data) == 0:
    st.warning("⚠️ The uploaded file contains 0 participants. Please upload a valid candidate pool.")
    st.stop()

# Rule 3: Integrate with scoring engine (write parsed JSON and run pipeline using hash cache)
# Incorporate the file contents and customized job criteria into the cache signature
config_signature = f"{job_title_input}_{','.join(sorted(required_toolkit))}"
state_key_raw = file_bytes + config_signature.encode('utf-8')
file_hash = hashlib.md5(state_key_raw).hexdigest()

if "last_processed_hash" not in st.session_state or st.session_state["last_processed_hash"] != file_hash:
    with st.spinner("Processing matching layer and ranking hybrid weights..."):
        try:
            # Sync the updated criteria with config.json first
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    cfg['target_job']['job_title'] = job_title_input
                    cfg['target_job']['required_skills'] = required_toolkit
                    cfg['target_job']['job_description'] = f"Seeking a highly skilled expert in {', '.join(required_toolkit)} for the role of {job_title_input}."
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
                except Exception as ex:
                    st.warning(f"Failed to synchronize config.json: {ex}")

            # Save parsed json to active path for scoring engine
            with open(active_candidates_path, "w", encoding="utf-8") as f:
                json.dump(candidates_data, f, indent=2)
            
            # Trigger our end-to-end pipeline execution with command-line arguments using required_toolkit
            python_executable = sys.executable if sys.executable else "python3"
            pipeline_path = os.path.join(CURRENT_DIR, "main_pipeline.py")
            result = subprocess.run(
                [python_executable, pipeline_path, "--job", job_title_input, "--skills", ",".join(required_toolkit)],
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

# Load calculated rankings from CSV output with Unicode fault-tolerance
leaderboard_csv = os.path.join(CURRENT_DIR, "data", "ranked_leaderboard.csv")
if os.path.exists(leaderboard_csv):
    try:
        try:
            df = pd.read_csv(leaderboard_csv)
        except UnicodeDecodeError:
            df = pd.read_csv(leaderboard_csv, encoding="latin-1")
    except Exception as e:
        st.error(f"Error reading compiled leaderboard: {e}")
        st.stop()
else:
    st.error("Leaderboard file was not generated by the pipeline.")
    st.stop()

# Apply the strict conditional boundaries and actions to match backend logic perfectly
def get_tier_and_action(score):
    if score >= 0.65:
        return "EXCEPTIONAL", "Fast-track to technical interview immediately."
    elif score >= 0.45:
        return "RECOMMENDED", "Move forward to initial recruiter screening."
    elif score >= 0.25:
        return "CONSIDER", "Review closely or place in alternative pipelines."
    else:
        return "NOT_ALIGNED", "Reject profile."

if not df.empty and "final_match_score" in df.columns:
    results = df["final_match_score"].apply(get_tier_and_action)
    df["hiring_confidence"] = [r[0] for r in results]
    df["suggested_action"] = [r[1] for r in results]

# Helper function to dynamically auto-adjust column layouts and convert to Excel bytes natively
def convert_df_to_excel(dataframe):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Leaderboard Results')
        # Apply standard auto-fit layout directly on properties
        workbook = writer.book
        worksheet = writer.sheets['Leaderboard Results']
        worksheet.views.sheetView[0].showGridLines = True
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)
    return output.getvalue()

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
    
    colors_map = {
        "EXCEPTIONAL": "#2ecc71",   # Emerald Green
        "RECOMMENDED": "#3498db",   # Bright Blue
        "CONSIDER": "#f1c40f",      # Amber/Yellow
        "NOT_ALIGNED": "#e74c3c"    # Crimson Red
    }

    with col_chart1:
        st.markdown("**Core Fit Score & Recommendation Distribution**")
        if "hiring_confidence" in df.columns:
            conf_counts = df["hiring_confidence"].value_counts().reset_index()
            conf_counts.columns = ["hiring_confidence", "Count"]
            fig_pie = px.pie(
                conf_counts, 
                names="hiring_confidence", 
                values="Count", 
                hole=0.45,
                color="hiring_confidence",
                color_discrete_map=colors_map,
                category_orders={"hiring_confidence": ["EXCEPTIONAL", "RECOMMENDED", "CONSIDER", "NOT_ALIGNED"]}
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
            df_top_10["anonymized_name"] = df_top_10["name"]
            df_top_10["fit_score"] = df_top_10["final_match_score"]
            
            # Dynamic categorization based strictly on actual score
            def assign_tier(score):
                if score >= 0.65: return "EXCEPTIONAL"
                elif score >= 0.45: return "RECOMMENDED"
                elif score >= 0.25: return "CONSIDER"
                else: return "NOT_ALIGNED"
            
            df_top_10['hiring_confidence'] = df_top_10['fit_score'].apply(assign_tier)
            
            # Sort dataframe descending by score so high scores render at the top of the Y-axis
            df_top_10 = df_top_10.sort_values(by="fit_score", ascending=False)
            
            colors_map = {
                "EXCEPTIONAL": "#2ecc71",   # Emerald Green
                "RECOMMENDED": "#3498db",   # Bright Blue
                "CONSIDER": "#f1c40f",      # Amber/Yellow
                "NOT_ALIGNED": "#e74c3c"    # Crimson Red
            }
            
            fig_bar = px.bar(
                df_top_10,
                x="fit_score",
                y="anonymized_name",
                color="hiring_confidence",
                color_discrete_map=colors_map,
                category_orders={"hiring_confidence": ["EXCEPTIONAL", "RECOMMENDED", "CONSIDER", "NOT_ALIGNED"]},
                orientation="h",
                labels={"fit_score": "Composite Core Score", "anonymized_name": "Candidate"},
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
                yaxis=dict(showgrid=False, automargin=True, autorange="reversed"),
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
        elif tier == "CONSIDER":
            badge_cls = "badge-caution"
        elif tier == "NOT_ALIGNED":
            badge_cls = "badge-notaligned"
            
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
            st.columns_list = st.columns(5)
            sc1, sc2, sc3, sc4, sc5 = st.columns_list
            
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

    # 5. Export calculated results as submission (Consolidated & Streamlined Layout)
    st.subheader("💾 Export Dynamic Analysis & Submissions")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Full technical background breakdown backup for engineering reference
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Rich Analysis (CSV)",
            data=csv_data,
            file_name="ranked_leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_d2:
        # NEW INTEGRATED FUNCTIONALITY: Generate official submission file automatically in .xlsx format
        excel_bytes = convert_df_to_excel(df)
        st.download_button(
            label="🟢 Download Official Submission (XLSX)",
            data=excel_bytes,
            file_name="Protocol_Zero_Final_Leaderboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )