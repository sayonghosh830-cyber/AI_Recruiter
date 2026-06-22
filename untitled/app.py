import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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
st.caption("E2E Prioritization based on skill overlap, semantic similarity, career stability, and recruitment signals")

# Side bar controls
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    # File upload handling
    uploaded_file = st.file_uploader("Upload Candidates Data (JSON or CSV)", type=["json", "csv"])
    
    use_default = st.checkbox("Use built-in candidate catalog if none uploaded", value=True)
    
    # Skill thresholds override info
    st.subheader("💡 Job Parameter Reference")
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            st.markdown(f"**Target Role:** `{cfg['target_job']['job_title']}`")
            st.markdown(f"**Required Toolkit:**")
            skills_md = " ".join([f"`{s}`" for s in cfg['target_job']['required_skills']])
            st.markdown(skills_md)
        except Exception as e:
            st.sidebar.error(f"Error reading job parameters: {e}")
            
    rank_btn = st.button("🚀 Rank Candidates", use_container_width=True, type="primary")

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Helper to map uploaded CSV to expected candidate JSON pipeline schema
def safe_str(val, default=""):
    if pd.isna(val) or val is None:
        return default
    s = str(val).strip()
    if s.lower() == 'nan':
        return default
    return s

def convert_csv_to_candidates_json(df_csv):
    candidates = []
    
    # Pre-clean columns to make search robust (case insensitive and space-free matching)
    col_mapping = {str(col).lower().replace(" ", "").replace("_", ""): col for col in df_csv.columns}
    
    def get_col_val(row, keys_list, default_val=None):
        for k in keys_list:
            norm_k = k.lower().replace(" ", "").replace("_", "")
            if norm_k in col_mapping:
                return row[col_mapping[norm_k]]
        return default_val

    for idx, row in df_csv.iterrows():
        # Ensure we always get a valid CAND_XXXXXXX ID
        raw_cand_id = safe_str(get_col_val(row, ['candidate_id', 'id', 'candidateid', 'cand_id']))
        digits_only = "".join([c for c in raw_cand_id if c.isdigit()])
        if not digits_only or len(digits_only) < 3:
            digits_only = str(1000000 + idx)
        cand_id = f"CAND_{digits_only[-7:].zfill(7)}"
        
        name = safe_str(get_col_val(row, ['name', 'fullname', 'anonymizedname', 'candidate_name']), f"Candidate {100+idx}")
        headline = safe_str(get_col_val(row, ['headline', 'currenttitle', 'title', 'role']), "Software Engineer")
        summary = safe_str(get_col_val(row, ['summary', 'description', 'about', 'bio']), f"Experienced developer. Background is: {headline}.")
        location = safe_str(get_col_val(row, ['location', 'city', 'address']), "Global")
        country = safe_str(get_col_val(row, ['country', 'nation']), "IN")
        
        try:
            raw_yoe = get_col_val(row, ['years_of_experience', 'yearsofexperience', 'experienceyears', 'experience', 'yoe', 'total_exp'])
            yoe = float(raw_yoe) if pd.notna(raw_yoe) and raw_yoe is not None else 3.0
            if pd.isna(yoe): yoe = 3.0
        except Exception:
            yoe = 3.0
        yoe = max(0.0, min(50.0, float(yoe)))
        
        company = safe_str(get_col_val(row, ['currentcompany', 'company', 'organization']), "N/A")
        valid_sizes = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"]
        comp_size = safe_str(get_col_val(row, ['currentcompanysize', 'companysize']), "51-200")
        if comp_size not in valid_sizes:
            comp_size = "51-200"
            
        industry = safe_str(get_col_val(row, ['currentindustry', 'industry', 'sector']), "Tech")
        
        # Parse comma/semicolon/pipe separated skills list
        raw_skills = safe_str(get_col_val(row, ['skills', 'skillslist', 'techstack', 'technologies', 'skills_list']))
        skills_list = []
        if raw_skills:
            delimiters = [',', ';', '|']
            skills_split = [raw_skills]
            for d in delimiters:
                if d in raw_skills:
                    skills_split = raw_skills.split(d)
                    break
            for s_name in skills_split:
                s_name_clean = s_name.strip()
                if s_name_clean:
                    skills_list.append({
                        "name": s_name_clean,
                        "proficiency": "expert" if yoe >= 5 else "intermediate",
                        "endorsements": 5,
                        "duration_months": int(yoe * 12) if yoe > 0 else 12
                    })
        if not skills_list:
            # Sane default skills
            skills_list = [{"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 24}]
            
        # Extract notice period
        notice_val = get_col_val(row, ['noticeperioddays', 'noticeperiod', 'notice_period_days'])
        try:
            if pd.isna(notice_val) or notice_val is None:
                notice_days = 30
            else:
                notice_days = int(float(notice_val))
            notice_days = max(0, min(180, notice_days))
        except Exception:
            notice_days = 30
            
        # Extract offer acceptance rate
        offer_val = get_col_val(row, ['offeracceptancerate', 'acceptancerate', 'offer_acceptance_rate'])
        try:
            if pd.isna(offer_val) or offer_val is None:
                offer_rate = 0.85
            else:
                offer_rate = float(offer_val)
            offer_rate = max(-1.0, min(1.0, offer_rate))
        except Exception:
            offer_rate = 0.85
            
        # Construct career history array to satisfy validation schema
        career_history = [{
            "company": company,
            "title": headline,
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": int(yoe * 12) if yoe > 0 else 12,
            "is_current": True,
            "industry": industry,
            "company_size": comp_size,
            "description": summary
        }]
        
        # Construct education history array to satisfy validation schema
        education = [{
            "institution": safe_str(get_col_val(row, ['university', 'institution', 'college', 'school']), "Unknown University"),
            "degree": safe_str(get_col_val(row, ['degree', 'qualification']), "B.S."),
            "field_of_study": safe_str(get_col_val(row, ['fieldofstudy', 'field', 'branch']), "Computer Science"),
            "start_year": 2018,
            "end_year": 2022,
            "grade": "3.5",
            "tier": "tier_2"
        }]
        
        # Build strict Schema Compliant Redrob Signals block
        try:
            profile_completeness_score = float(get_col_val(row, ['profilecompletenessscore', 'profilecompleteness', 'completeness'], 85.0))
            if pd.isna(profile_completeness_score): profile_completeness_score = 85.0
            profile_completeness_score = max(0.0, min(100.0, profile_completeness_score))
        except Exception:
            profile_completeness_score = 85.0
            
        try:
            recruiter_response_rate = float(get_col_val(row, ['recruiterresponserate', 'responserate'], 0.90))
            if pd.isna(recruiter_response_rate): recruiter_response_rate = 0.90
            recruiter_response_rate = max(0.0, min(1.0, recruiter_response_rate))
        except Exception:
            recruiter_response_rate = 0.90
            
        try:
            interview_completion_rate = float(get_col_val(row, ['interviewcompletionrate', 'interviewcompletion'], 0.95))
            if pd.isna(interview_completion_rate): interview_completion_rate = 0.95
            interview_completion_rate = max(0.0, min(1.0, interview_completion_rate))
        except Exception:
            interview_completion_rate = 0.95
            
        try:
            conn_count = int(float(get_col_val(row, ['connectioncount', 'connections'], 120)))
            conn_count = max(0, conn_count)
        except Exception:
            conn_count = 120
            
        try:
            endorse_received = int(float(get_col_val(row, ['endorsementsreceived', 'endorsements'], 15)))
            endorse_received = max(0, endorse_received)
        except Exception:
            endorse_received = 15
            
        try:
            views_30d = int(float(get_col_val(row, ['profileviewsreceived30d', 'profileviews', 'views'], 12)))
            views_30d = max(0, views_30d)
        except Exception:
            views_30d = 12
            
        try:
            apps_30d = int(float(get_col_val(row, ['applicationssubmitted30d', 'applications', 'applicants'], 5)))
            apps_30d = max(0, apps_30d)
        except Exception:
            apps_30d = 5
            
        try:
            github_score = float(get_col_val(row, ['githubactivityscore', 'githubscore', 'github'], 65.0))
            if pd.isna(github_score): github_score = 65.0
            github_score = max(-1.0, min(100.0, github_score))
        except Exception:
            github_score = 65.0
            
        try:
            search_apps = int(float(get_col_val(row, ['searchappearance30d', 'searchappearances', 'searches'], 45)))
            search_apps = max(0, search_apps)
        except Exception:
            search_apps = 45
            
        try:
            saved_rec = int(float(get_col_val(row, ['savedbyrecruiters30d', 'savedbyrecruiters', 'saved'], 8)))
            saved_rec = max(0, saved_rec)
        except Exception:
            saved_rec = 8
            
        try:
            resp_time = float(get_col_val(row, ['avgresponsetimehours', 'responsetime', 'response_time'], 4.5))
            if pd.isna(resp_time): resp_time = 4.5
            resp_time = max(0.0, resp_time)
        except Exception:
            resp_time = 4.5
            
        try:
            salary_min = float(get_col_val(row, ['expectedsalaryminlpa', 'salarymin', 'minsalary'], 12.0))
            if pd.isna(salary_min): salary_min = 12.0
            salary_min = max(0.0, salary_min)
        except Exception:
            salary_min = 12.0
            
        try:
            salary_max = float(get_col_val(row, ['expectedsalarymaxlpa', 'salarymax', 'maxsalary'], 24.0))
            if pd.isna(salary_max): salary_max = max(salary_min, 24.0)
            salary_max = max(0.0, salary_max)
        except Exception:
            salary_max = 24.0
            
        redrob_signals = {
            "profile_completeness_score": float(profile_completeness_score),
            "signup_date": "2023-01-01",
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True if str(get_col_val(row, ['opentowork', 'opentoworkflag', 'open_to_work'], 'True')).lower() == 'true' else False,
            "profile_views_received_30d": int(views_30d),
            "applications_submitted_30d": int(apps_30d),
            "recruiter_response_rate": float(recruiter_response_rate),
            "avg_response_time_hours": float(resp_time),
            "skill_assessment_scores": {},
            "connection_count": int(conn_count),
            "endorsements_received": int(endorse_received),
            "notice_period_days": int(notice_days),
            "expected_salary_range_inr_lpa": {
                "min": float(salary_min),
                "max": float(salary_max)
            },
            "preferred_work_mode": str(get_col_val(row, ['preferredworkmode', 'workmode', 'preferred_work_mode'], 'hybrid')).lower() if str(get_col_val(row, ['preferredworkmode', 'workmode', 'preferred_work_mode'], 'hybrid')).lower() in ["remote", "hybrid", "onsite", "flexible"] else "hybrid",
            "willing_to_relocate": True if str(get_col_val(row, ['willingtorelocate', 'willing_to_relocate'], 'True')).lower() == 'true' else False,
            "github_activity_score": float(github_score),
            "search_appearance_30d": int(search_apps),
            "saved_by_recruiters_30d": int(saved_rec),
            "interview_completion_rate": float(interview_completion_rate),
            "offer_acceptance_rate": float(offer_rate),
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
        
        profile = {
            "candidate_id": cand_id,
            "profile": {
                "anonymized_name": name,
                "headline": headline,
                "summary": summary,
                "location": location,
                "country": country,
                "years_of_experience": float(yoe),
                "current_title": headline,
                "current_company": company,
                "current_company_size": comp_size,
                "current_industry": industry
            },
            "career_history": career_history,
            "education": education,
            "skills": skills_list,
            "redrob_signals": redrob_signals
        }
        candidates.append(profile)
    return candidates

# Validations for user uploaded candidate profiles
def validate_candidates_json_data(data):
    if not isinstance(data, list):
        return False, "Data must be a list of candidate profiles JSON arrays."
    if len(data) == 0:
        return False, "Data list is empty."
    first_item = data[0]
    if not isinstance(first_item, dict):
        return False, "Each candidate item must be a JSON object containing profile details."
    # Check if this item is a submission output instead of profile
    submission_keys = {"rank", "score", "reasoning"}
    first_keys = set(first_item.keys())
    if submission_keys.issubset(first_keys):
        return False, "This resembles a submission results CSV formatted as JSON (contains rank, score, reasoning) instead of raw candidate profile objects."
    valid_candidate_keys = {"profile", "skills", "redrob_signals"}
    has_valid_keys = any(k in first_item for k in valid_candidate_keys)
    if not has_valid_keys:
        return False, "The file does not match Redrob candidate profile structure (it lacks profile, skills, or redrob_signals schema)."
    return True, ""

def validate_candidates_csv_data(df_csv):
    cols = [str(c).lower().replace(" ", "").replace("_", "") for c in df_csv.columns]
    mandatory_sub_cols = {"rank", "score", "reasoning"}
    if all(x in cols for x in mandatory_sub_cols):
        return False, "The uploaded file is a Submission output template (contains rank, score, reasoning). Please upload raw candidate profiles."
    valid_cand_indicators = {"name", "fullname", "anonymizedname", "candidateid", "skills", "headline", "currenttitle", "role", "experience", "yoe", "years_of_experience"}
    if not any(indicator in cols for x in cols for indicator in valid_cand_indicators if indicator in x):
        return False, "The uploaded CSV lacks expected candidate columns (such as name, skills, headline, or years of experience)."
    return True, ""

# Safe paths assignment
pristine_candidates_path = "data/sample_candidates.json"
active_candidates_path = "data/active_candidates.json"

# Default fallback if no file uploaded
if uploaded_file is None:
    if os.path.exists(pristine_candidates_path):
        try:
            import shutil
            shutil.copy(pristine_candidates_path, active_candidates_path)
        except Exception as e:
            st.sidebar.error(f"Error restoring default candidate catalog: {e}")
else:
    try:
        if uploaded_file.name.endswith(".json"):
            candidates_data = json.load(uploaded_file)
            is_valid, err_msg = validate_candidates_json_data(candidates_data)
            if is_valid:
                with open(active_candidates_path, "w", encoding="utf-8") as f:
                    json.dump(candidates_data, f, indent=2)
                st.sidebar.success(f"Successfully validated & loaded {len(candidates_data)} JSON candidate profiles!")
            else:
                st.sidebar.error(f"⚠️ Invalid Candidate Profiles: {err_msg}")
                # Fallback to default copy to keep workspace functional
                if os.path.exists(pristine_candidates_path):
                    import shutil
                    shutil.copy(pristine_candidates_path, active_candidates_path)
        elif uploaded_file.name.endswith(".csv"):
            df_csv = pd.read_csv(uploaded_file)
            is_valid, err_msg = validate_candidates_csv_data(df_csv)
            if is_valid:
                candidates_data = convert_csv_to_candidates_json(df_csv)
                with open(active_candidates_path, "w", encoding="utf-8") as f:
                    json.dump(candidates_data, f, indent=2)
                st.sidebar.success(f"Mapped {len(candidates_data)} CSV candidates to active pipeline layout!")
            else:
                st.sidebar.error(f"⚠️ Invalid Candidate Profiles: {err_msg}")
                # Fallback to default copy to keep workspace functional
                if os.path.exists(pristine_candidates_path):
                    import shutil
                    shutil.copy(pristine_candidates_path, active_candidates_path)
    except Exception as e:
        st.sidebar.error(f"Failed to process uploaded file: {e}")

# Check outputs paths
leaderboard_csv = "data/ranked_leaderboard.csv"

# Run Pipeline on click
if rank_btn:
    with st.spinner("Processing matching layer and ranking hybrid weights..."):
        try:
            # Multi-OS execution support utilizing current runtime context executable
            python_executable = sys.executable if sys.executable else "python3"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pipeline_path = os.path.join(current_dir, "main_pipeline.py")
            result = subprocess.run(
                [python_executable, pipeline_path],
                cwd=current_dir,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                st.sidebar.info("Console Log:\n" + result.stdout[-300:])
            st.success("Pipeline executed successfully! Leaderboard compiled.")
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline execution failed: {e}")
            if e.stderr:
                st.code(e.stderr)
            if e.stdout:
                st.sidebar.info("Console Log:\n" + e.stdout[-300:])
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")

# Load existing or generated leaderboard
if os.path.exists(leaderboard_csv):
    try:
        df = pd.read_csv(leaderboard_csv)
    except Exception as e:
        st.error(f"Error reading compiled leaderboard: {e}")
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

# If spreadsheet is loaded, render metrics and analysis
if not df.empty:
    # 3. KPI Cards Column Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_profiles = len(df)
    top_score = df["final_match_score"].max() if "final_match_score" in df.columns else 0.0
    avg_score = df["final_match_score"].mean() if "final_match_score" in df.columns else 0.0
    
    rec_count = 0
    if "hiring_confidence" in df.columns:
        rec_count = len(df[df["hiring_confidence"].isin(["EXCEPTIONAL", "RECOMMENDED"])])
        
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

    # 4. Interactive Plotly Visualizations Section
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
            df_top_10 = df_top_10.iloc[::-1] # horizontal bar displays best on top when reversed
            
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
                xaxis=dict(showgrid=True, gridcolor="#30363d", range=[0, 1.0]),
                yaxis=dict(showgrid=False),
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Score Distribution Histogram
    st.markdown("**Score Density Profile (Interactive Cohort Distribution)**")
    if "final_match_score" in df.columns:
        fig_hist = px.histogram(
            df,
            x="final_match_score",
            nbins=15,
            color_discrete_sequence=["#58a6ff"],
            labels={"final_match_score": "Composite Fit Score"}
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#c9d1d9",
            xaxis=dict(showgrid=True, gridcolor="#30363d"),
            yaxis=dict(showgrid=True, gridcolor="#30363d"),
            margin=dict(t=10, b=15, l=10, r=10),
            bargap=0.1
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # 5. Top 25 Dataframe table
    st.subheader("🏆 Live Leaderboard (Top 25)")
    df_top = df.head(25)
    
    display_cols = ["leaderboard_rank", "candidate_id", "name", "final_match_score", "hiring_confidence", "suggested_action"]
    valid_cols = [c for c in display_cols if c in df_top.columns]
    
    has_matplotlib = False
    try:
        import matplotlib
        has_matplotlib = True
    except ImportError:
        pass

    styled_df = df_top[valid_cols]
    if has_matplotlib:
        try:
            styled_df = df_top[valid_cols].style.background_gradient(subset=["final_match_score"], cmap="Blues")
        except Exception:
            styled_df = df_top[valid_cols]
        
    st.dataframe(
        styled_df,
        use_container_width=True
    )

    # 6. Expandable Candidate Cards
    st.subheader("🔍 Selected Candidate Summaries & Evaluation Explanations")
    
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
            
            # Show additional detail signals loaded securely with robust pandas fallbacks
            st.markdown("**🔬 Metric Diagnostic Profile:**")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            
            stab = row.get("stability_score", row.get("tenure_stability_score", 0.0))
            try:
                stab_val = f"{float(stab):.2f}" if pd.notna(stab) else "0.00"
            except Exception:
                stab_val = "0.00"
            sc1.metric("Tenure Stability", stab_val)
            
            acad = row.get("academic_prestige", row.get("academic_prestige_score", 0.0))
            try:
                acad_val = f"{float(acad):.2f}" if pd.notna(acad) else "0.00"
            except Exception:
                acad_val = "0.00"
            sc2.metric("Academic Prestige", acad_val)
            
            resp = row.get("platform_responsiveness", row.get("responsiveness_score", 0.0))
            try:
                resp_val = f"{float(resp):.2f}" if pd.notna(resp) else "0.00"
            except Exception:
                resp_val = "0.00"
            sc3.metric("Platform Responsiveness", resp_val)
            
            attend = row.get("attendance_reliability", row.get("reliability_score", 0.0))
            try:
                attend_val = f"{float(attend):.2f}" if pd.notna(attend) else "0.00"
            except Exception:
                attend_val = "0.00"
            sc4.metric("Attendance Reliability", attend_val)
            
            notice = row.get("notice_period_days", row.get("notice_period", "Immediate"))
            try:
                notice_str = f"{int(float(notice))} Days" if pd.notna(notice) and str(notice).strip() != '' else "Immediate"
            except Exception:
                notice_str = str(notice) if pd.notna(notice) else "Immediate"
            sc5.metric("Notice Period", notice_str)

    # 7. Download CSV buttons in primary sidebar and main page
    st.subheader("💾 Export Leaderboard & Submissions")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Rich Analysis Dashboard (CSV)",
            data=csv_data,
            file_name="ranked_leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_d2:
        sub_file = Path("data/submission.csv")
        if sub_file.exists():
            try:
                with open(sub_file, "r", encoding="utf-8") as f:
                    sub_csv_text = f.read()
                st.download_button(
                    label="🎯 Download Platform Submission File (submission.csv)",
                    data=sub_csv_text.encode('utf-8'),
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception:
                sub_file = None
        
        if sub_file is None or not Path("data/submission.csv").exists():
            try:
                # Fallback if submission.csv hasn't been generated yet, we construct it on-the-fly
                fallback_sub = pd.DataFrame()
                fallback_sub["candidate_id"] = df["candidate_id"]
                fallback_sub["rank"] = df["leaderboard_rank"]
                fallback_sub["score"] = df["final_match_score"]
                fallback_sub["reasoning"] = df.get("one_liner_reasoning", "No explicit reasoning details loaded.")
                st.download_button(
                    label="🎯 Download Platform Submission File (submission.csv)",
                    data=fallback_sub.to_csv(index=False).encode('utf-8'),
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception:
                st.write("Process candidates to unlock direct challenge submission.")
else:
    st.info("👋 Welcome! Click the **🚀 Rank Candidates** button in the sidebar to execute the hybrid prioritization pipeline.")
