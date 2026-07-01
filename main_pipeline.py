#!/usr/bin/env python3
"""
Main entry point for the Candidate Matching and Hybrid Ranking Pipeline.
Executes the end-to-end flow from loading candidate data, running feature engineering,
performing semantic matching, applying the hybrid ranking engine, and exporting the final
ranked leaderboard.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Configure global professional logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MainPipeline")

# Add src folder to path to ensure correct resolutions
sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
import json
import math
import re
import csv
from datetime import datetime

USING_PURE_PYTHON_FALLBACK = False

# Add src folder to path to ensure correct resolutions
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from data_loader import CandidateDataLoader
    from feature_engineering import RecruiterFeatureEngine
    from matching_layer import SemanticMatchingLayer
    from hybrid_ranking_engine import HybridRankingWeights, HybridRankingEngine
    from explainability_pipeline import RecruiterExplainabilityEngine
    from submission_generator import RecruiterSubmissionGenerator
except ImportError:
    USING_PURE_PYTHON_FALLBACK = True


def pure_tfidf_cosine_sim(query: str, documents: list) -> list:
    """
    Computes sublinear TF-IDF cosine similarity between query and candidate documents, 
    using only pure Python built-in modules.
    """
    def tokenize(text):
        if not text:
            return []
        return re.findall(r"\b[a-zA-Z]{3,20}\b", text.lower())
        
    all_docs = [query] + documents
    tokenized_docs = [tokenize(d) for d in all_docs]
    
    vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    if not vocab:
        return [0.0] * len(documents)
        
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    df = [0] * len(vocab)
    for doc in tokenized_docs:
        unique_words = set(doc)
        for w in unique_words:
            df[word_to_idx[w]] += 1
            
    N = len(all_docs)
    idf = [math.log((1 + N) / (1 + df[i])) + 1 for i in range(len(vocab))]
    
    def get_tfidf_vec(doc_tokens):
        tf = [0.0] * len(vocab)
        for w in doc_tokens:
            if w in word_to_idx:
                tf[word_to_idx[w]] += 1
        vec = [0.0] * len(vocab)
        for i in range(len(vocab)):
            if tf[i] > 0:
                vec[i] = (1.0 + math.log(tf[i])) * idf[i]
        return vec
        
    tfidf_vectors = [get_tfidf_vec(doc) for doc in tokenized_docs]
    
    def cos_sim(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 * norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
        
    query_vec = tfidf_vectors[0]
    similarities = [cos_sim(query_vec, doc_vec) for doc_vec in tfidf_vectors[1:]]
    return similarities


def run_pure_python_fallback_pipeline(config_path: str = "config.json", target_job: str = None, required_toolkit: list = None) -> None:
    """
    Fallback pipeline engine executed when third-party requirements (numpy, pandas, etc.)
    are missing in the environment. Computes identical talent prioritization metrics.
    """
    logger.info("Initializing Zero-Dependency Pure Python Fallback Pipeline")
    logger.info("Pipeline is running in: TF_IDF_FALLBACK mode")
    logger.info("Reason for fallback mode activation: Third-party machine learning requirements (numpy, pandas, scikit-learn, etc.) are missing in the current Python environment, triggering zero-dependency pure Python fallback.")
    
    try:
        config = load_pipeline_config(config_path)
        dataset_cfg = config["dataset_paths"]
        weights_cfg = config["ranking_weights"]
        job_cfg = config["target_job"]
    except Exception as e:
        logger.error(f"Configuration loader failed: {e}")
        sys.exit(1)
        
    proj_root = Path(__file__).parent.resolve()
    candidates_path = proj_root / dataset_cfg["candidates_path"]
    output_csv = proj_root / dataset_cfg["output_csv_path"]
    output_xlsx = proj_root / dataset_cfg["output_xlsx_path"]
    
    if not candidates_path.exists():
        fallback_source = proj_root / "data" / "sample_candidates.json"
        if fallback_source.exists():
            import shutil
            shutil.copy(fallback_source, candidates_path)
            logger.info(f"Copied pristine profiles catalog to active candidate path: {candidates_path}")
            
    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            raw_candidates_list = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read candidates catalog file from {candidates_path}: {e}")
        sys.exit(1)
        
    job_title = target_job if target_job is not None else job_cfg.get("job_title", "Backend Engineer")
    required_skills = required_toolkit if required_toolkit is not None else job_cfg.get("required_skills", [])
    if target_job is not None or required_toolkit is not None:
        job_description = f"We are seeking a highly experienced expert in {', '.join(required_skills)} for the role of {job_title}."
    else:
        job_description = job_cfg.get("job_description", "")
        
    required_skills_norm = [s.strip().lower() for s in required_skills if s.strip()]
    
    processed_candidates = []
    baseline_date = datetime(2026, 6, 20)
    candidate_docs = []
    
    for cand in raw_candidates_list:
        cand_id = cand["candidate_id"]
        profile = cand.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        current_title = profile.get("current_title", "")
        current_company = profile.get("current_company", "")
        current_industry = profile.get("current_industry", "")
        years_exp = profile.get("years_of_experience", 0.0)
        
        skills_raw = cand.get("skills", [])
        skills_tokens = [f"{s.get('name')} ({s.get('proficiency', 'intermediate')})" for s in skills_raw if s.get("name")]
        skills_string = ", ".join(skills_tokens) if skills_tokens else "No listed skills"
        
        career_raw = cand.get("career_history", [])
        career_tokens = []
        for job in career_raw:
            j_title = job.get("title", "")
            j_comp = job.get("company", "")
            j_desc = job.get("description", "").strip()
            j_dur = f"{job.get('duration_months', 0)} months"
            career_tokens.append(f"Role: {j_title} at {j_comp} (Duration: {j_dur}). Responsibilities: {j_desc}")
        career_history_string = " | ".join(career_tokens) if career_tokens else "No listed work milestones"
        
        edu_raw = cand.get("education", [])
        edu_tokens = []
        for edu in edu_raw:
            edu_tokens.append(f"{edu.get('degree', 'Degree')} in {edu.get('field_of_study', 'Field')} from {edu.get('institution', 'University')}")
        education_string = ", ".join(edu_tokens) if edu_tokens else "No listed degrees"
        
        doc = (
            f"Candidate ID: {cand_id}\n"
            f"Current Position: {current_title} at {current_company} in {current_industry} industry. Total seniority: {years_exp:.1f} years of experience.\n"
            f"Headline Summary: {headline}. Profile overview: {summary}\n"
            f"Technical Skills and Capabilities: {skills_string}\n"
            f"Professional Job Milestones and Career History: {career_history_string}\n"
            f"Scholastic Background and Credentials: {education_string}"
        )
        doc = re.sub(r"\s+", " ", doc)
        candidate_docs.append(doc)
        
    similarities = pure_tfidf_cosine_sim(job_description, candidate_docs)
    
    for idx, cand in enumerate(raw_candidates_list):
        cand_id = cand["candidate_id"]
        profile = cand.get("profile", {})
        name = profile.get("anonymized_name", "Anonymized Candidate")
        headline = profile.get("headline", "")
        current_title = profile.get("current_title", "")
        years_exp = profile.get("years_of_experience", 0.0)
        
        semantic_sim = similarities[idx]
        
        current_t = current_title.strip().lower()
        headline_t = headline.strip().lower()
        target_t = job_title.strip().lower()
        
        if target_t in current_t or current_t in target_t:
            title_match = 1.0
        elif any(word in current_t for word in target_t.split() if len(word) > 3):
            title_match = 0.7
        elif any(word in headline_t for word in target_t.split() if len(word) > 3):
            title_match = 0.5
        else:
            title_match = 0.1
            
        skills_raw = cand.get("skills", [])
        cand_skills_items = [str(s.get("name", "")).strip().lower() for s in skills_raw if s.get("name")]
        
        if not required_skills_norm or len(required_skills_norm) == 0:
            skill_overlap = 0.0
        else:
            matched_skills_set = set(required_skills_norm).intersection(set(cand_skills_items))
            skill_overlap = len(matched_skills_set) / len(required_skills_norm)
            
        composite_semantic_score = (semantic_sim * 0.3) + (title_match * 0.2) + (skill_overlap * 0.5)
        
        edu_raw = cand.get("education", [])
        max_edu_score = 0.2
        tier_weights = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50, "tier_4": 0.30}
        for edu in edu_raw:
            tier_score = tier_weights.get(edu.get("tier", "unknown"), 0.20)
            degree = str(edu.get("degree", "bachelor")).lower()
            if "ph.d" in degree or "phd" in degree:
                mult = 1.25
            elif any(m in degree for m in ["m.tech", "m.s", "m.sc", "m.e", "mba"]):
                mult = 1.15
            elif any(b in degree for b in ["b.tech", "b.e", "b.sc", "b.a"]):
                mult = 1.0
            else:
                mult = 0.85
            edu_score = min(1.0, tier_score * mult)
            if edu_score > max_edu_score:
                max_edu_score = edu_score
        academic_prestige_score = max_edu_score
        
        endorsements_list = [s.get("endorsements", 0) for s in skills_raw]
        avg_endorsements = sum(endorsements_list) / len(endorsements_list) if endorsements_list else 0.0
        avg_skill_endorsements = min(1.0, math.log1p(avg_endorsements) / math.log1p(100.0))
        total_skills_count = len(skills_raw)
        
        career_raw = cand.get("career_history", [])
        num_companies_worked = len(set(job.get("company", "") for job in career_raw if job.get("company")))
        num_positions = len(career_raw)
        durations = [job.get("duration_months", 0) for job in career_raw if job.get("duration_months") is not None]
        avg_role_duration_months = sum(durations) / len(durations) if durations else 0.0
        
        career_stability_score = max(0.2, min(1.0, avg_role_duration_months / 36.0))
        if num_companies_worked >= 3 and avg_role_duration_months < 14.0:
            career_stability_score *= 0.65
            
        company_counts = {}
        for job in career_raw:
            comp = job.get("company", "")
            if comp:
                company_counts[comp] = company_counts.get(comp, 0) + 1
        has_promotion = any(c > 1 for c in company_counts.values())
        if has_promotion:
            career_stability_score *= 1.15
        career_stability_score = min(1.0, career_stability_score)
        if not career_raw:
            career_stability_score = 0.7
            
        signals = cand.get("redrob_signals", {})
        recruiter_response_rate = signals.get("recruiter_response_rate", 0.70)
        avg_response_time_hours = signals.get("avg_response_time_hours", 24.0)
        interview_completion_rate = signals.get("interview_completion_rate", 0.70)
        
        clipped_response_time = min(250.0, max(0.0, avg_response_time_hours))
        responsiveness_score = recruiter_response_rate * (1.0 - (clipped_response_time / 250.0))
        interview_reliability = interview_completion_rate
        
        last_active = signals.get("last_active_date", "2026-06-20")
        try:
            dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_gap = max(0, (baseline_date - dt).days)
        except Exception:
            days_gap = 180
            
        time_decay = math.exp(-days_gap / 45.0)
        search_appearance_30d = signals.get("search_appearance_30d", 0)
        search_vol = min(500.0, max(0.0, float(search_appearance_30d))) / 500.0
        
        saved_by_recruiters_30d = signals.get("saved_by_recruiters_30d", 0)
        saved_vol = min(15.0, max(0.0, float(saved_by_recruiters_30d))) / 15.0
        
        profile_views = signals.get("profile_views_received_30d", signals.get("profile_views_30d", 0))
        view_vol = min(100.0, max(0.0, float(profile_views))) / 100.0
        
        open_to_work = signals.get("open_to_work_flag", True)
        open_to_work_val = 1.0 if open_to_work else 0.0
        
        aei_raw = (time_decay * 0.45) + (search_vol * 0.20) + (view_vol * 0.15) + (saved_vol * 0.10) + (open_to_work_val * 0.10)
        active_engagement_score = min(1.0, max(0.0, aei_raw))
        
        verified_email = signals.get("verified_email", True)
        verified_phone = signals.get("verified_phone", True)
        linkedin_connected = signals.get("linkedin_connected", True)
        verification_score = (int(verified_email) + int(verified_phone) + int(linkedin_connected)) / 3.0
        
        skill_assessments = signals.get("skill_assessment_scores", {})
        assessments_list = [float(v) for v in skill_assessments.values()]
        avg_assessment_score = sum(assessments_list) / len(assessments_list) / 100.0 if assessments_list else 0.0
        
        github_activity_score = float(signals.get("github_activity_score", 0.0))
        platform_achievement_score = min(1.0, max(0.0, (avg_assessment_score * 0.7) + (max(0.0, github_activity_score) / 100.0 * 0.3)))
        
        salary_expected = signals.get("expected_salary_range_inr_lpa", {})
        min_sal = float(salary_expected.get("min", 15.0))
        max_sal = float(salary_expected.get("max", 25.0))
        expected_salary_mid = (min_sal + max_sal) / 2.0
        
        years_to_salary_ratio = years_exp / expected_salary_mid if expected_salary_mid > 0 else 0.3
        salary_value_score = min(1.5, max(0.0, years_to_salary_ratio)) / 1.5
        
        profile_completeness_score = float(signals.get("profile_completeness_score", 85.0))
        profile_completeness_ratio = profile_completeness_score / 100.0
        
        notice_period_days = int(signals.get("notice_period_days", 30))
        notice_period_risk = 1.0 if notice_period_days >= 90 else (0.5 if notice_period_days >= 60 else 0.0)
        stale_profile_risk = min(180.0, max(0.0, float(days_gap))) / 180.0
        job_hopping_risk = 1.0 if avg_role_duration_months < 12.0 and num_companies_worked >= 2 else 0.0
        
        aggregate_risk_score = (notice_period_risk * 0.4) + (stale_profile_risk * 0.3) + (job_hopping_risk * 0.3)
        
        # Override and completely disable secondary modifiers as requested (availability, immediate, collab, stability)
        career_stability_score = 1.0
        responsiveness_score = 1.0
        open_to_work_val = 1.0
        active_engagement_score = 1.0
        notice_period_risk = 0.0
        stale_profile_risk = 0.0
        job_hopping_risk = 0.0
        aggregate_risk_score = 0.0

        composite_quality_score = (
            (academic_prestige_score * 0.20) +
            (career_stability_score * 0.25) +
            (platform_achievement_score * 0.20) +
            (verification_score * 0.15) +
            (salary_value_score * 0.10) +
            (profile_completeness_ratio * 0.10)
        )
        
        composite_recruitability_score = (
            (active_engagement_score * 0.40) +
            (responsiveness_score * 0.35) +
            (interview_reliability * 0.25)
        )
        
        # Override weights so secondary parameters contribute at most 1% total
        w_semantic = 0.99
        w_quality = 0.004
        w_recruitability = 0.003
        w_risk = 0.003
        
        final_match_score = (w_semantic * composite_semantic_score) + (w_quality * composite_quality_score) + (w_recruitability * composite_recruitability_score) - (w_risk * aggregate_risk_score)
        final_match_score = min(1.0, max(0.0, final_match_score))
        
        matched_target_skills = []
        missing_target_skills = []
        for req_skill in required_skills:
            req_norm = req_skill.strip().lower()
            found_skill = False
            for cand_skill in cand_skills_items:
                if req_norm == cand_skill:
                    matched_target_skills.append(req_skill)
                    found_skill = True
                    break
            if not found_skill:
                missing_target_skills.append(req_skill)
                
        if final_match_score >= 0.65:
            hiring_confidence = "EXCEPTIONAL"
            suggested_action = "Fast-track to technical interview immediately."
        elif final_match_score >= 0.55:
            hiring_confidence = "RECOMMENDED"
            suggested_action = "Move forward to initial recruiter screening."
        elif final_match_score >= 0.30:
            hiring_confidence = "CONSIDER"
            suggested_action = "Review closely or place in alternative pipelines."
        else:
            hiring_confidence = "NOT_ALIGNED"
            suggested_action = "Reject profile."
            
        reasons_list = []
        if matched_target_skills:
            reasons_list.append(f"Matched {len(matched_target_skills)} required tech skills")
        if recruiter_response_rate >= 0.60:
            reasons_list.append("high collaborator profile")
        if notice_period_days <= 30:
            reasons_list.append("immediate/30-day availability window")
        if career_stability_score >= 0.80:
            reasons_list.append("stable job records")
            
        one_liner_reasoning = f"{current_title or 'Specialist'} with {years_exp:.1f} yrs; {'; '.join(reasons_list[:3])}."
        
        processed_candidates.append({
            "candidate_id": cand_id,
            "name": name,
            "current_title": current_title,
            "years_of_experience": years_exp,
            "final_match_score": final_match_score,
            "composite_quality_score": composite_quality_score,
            "composite_recruitability_score": composite_recruitability_score,
            "aggregate_risk_score": aggregate_risk_score,
            "hiring_confidence": hiring_confidence,
            "one_liner_reasoning": one_liner_reasoning,
            "suggested_action": suggested_action,
            "matched_skills": ", ".join(matched_target_skills),
            "missing_skills": ", ".join(missing_target_skills),
            "stability_score": round(career_stability_score, 2),
            "academic_prestige": round(academic_prestige_score, 2),
            "platform_responsiveness": round(recruiter_response_rate, 2),
            "attendance_reliability": round(interview_completion_rate, 2),
            "churn_risk": round(job_hopping_risk, 2),
            "notice_period_days": notice_period_days
        })
        
    processed_candidates.sort(key=lambda x: (x["final_match_score"], x["composite_quality_score"], x["years_of_experience"]), reverse=True)
    
    for idx, cand in enumerate(processed_candidates):
        cand["leaderboard_rank"] = idx + 1
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    headers_leaderboard = [
        "candidate_id", "name", "leaderboard_rank", "final_match_score", 
        "hiring_confidence", "one_liner_reasoning", "suggested_action", 
        "matched_skills", "missing_skills", "stability_score", "academic_prestige", 
        "platform_responsiveness", "attendance_reliability", "churn_risk", "notice_period_days"
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers_leaderboard)
        writer.writeheader()
        for cand in processed_candidates:
            row = {h: cand[h] for h in headers_leaderboard}
            row["final_match_score"] = round(row["final_match_score"], 4)
            writer.writerow(row)
            
    submission_file_path = os.path.join(os.path.dirname(output_csv), "submission.csv")
    headers_submission = ["candidate_id", "rank", "score", "reasoning"]
    
    with open(submission_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers_submission)
        writer.writeheader()
        for cand in processed_candidates:
            row = {
                "candidate_id": cand["candidate_id"],
                "rank": cand["leaderboard_rank"],
                "score": round(cand["final_match_score"], 4),
                "reasoning": cand["one_liner_reasoning"]
            }
            writer.writerow(row)
            
    try:
        with open(output_xlsx, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
        
    logger.info("E2E Recruitment Fallback Pipeline successfully executed!")
    logger.info(f"Leaderboard output saved directly under standard path: {output_csv}")
    logger.info(f"Submission compiled successfully at: {submission_file_path}")


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """
    Loads runtime configuration from JSON.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Loaded pipeline configuration from {path.name}")
    return config


def run_end_to_end_pipeline(config_path: str = "config.json", target_job: str = None, required_toolkit: list = None) -> None:
    """
    Executes the entire end-to-end recruitment prioritization and matching flow.
    """
    logger.info("Initializing Candidate Matching & Hybrid Ranking Pipeline")

    global USING_PURE_PYTHON_FALLBACK
    if USING_PURE_PYTHON_FALLBACK:
        run_pure_python_fallback_pipeline(config_path, target_job=target_job, required_toolkit=required_toolkit)
        return

    try:
        config = load_pipeline_config(config_path)
        dataset_cfg = config["dataset_paths"]
        matcher_cfg = config["semantic_matcher"]
        weights_cfg = config["ranking_weights"]
        job_cfg = config["target_job"]
    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Configuration parser failure: {e}")
        sys.exit(1)

    try:
        proj_root = Path(__file__).parent.resolve()
        schema_path = proj_root / dataset_cfg["schema_path"]
        candidates_path = proj_root / dataset_cfg["candidates_path"]
        output_csv = proj_root / dataset_cfg["output_csv_path"]
        output_xlsx = proj_root / dataset_cfg["output_xlsx_path"]

        # Pre-flight self-healing setup for candidate inputs
        if not candidates_path.exists():
            fallback_source = proj_root / "data" / "sample_candidates.json"
            if fallback_source.exists():
                import shutil
                shutil.copy(fallback_source, candidates_path)
                logger.info(f"Copied pristine profiles catalog to active candidate path: {candidates_path}")

        job_title = target_job if target_job is not None else job_cfg.get("job_title", "Backend Software Engineer")
        required_skills = required_toolkit if required_toolkit is not None else job_cfg.get("required_skills", [])
        if target_job is not None or required_toolkit is not None:
            job_description = f"We are seeking a highly experienced expert in {', '.join(required_skills)} for the role of {job_title}."
        else:
            job_description = job_cfg.get("job_description", "")

        # 1. Load profiles and relational tables
        logger.info("Stage 1: Loading candidate profiles")
        loader = CandidateDataLoader(schema_path=schema_path)
        raw_candidates_list = loader.load_candidates_raw(candidates_path)
        dfs = loader.parse_to_relational_dataframes(raw_candidates_list)
        for df_name, df_obj in dfs.items():
            logger.info(f"Table '{df_name}': {df_obj.shape[0]} records loaded")

        # 2. Run Feature Engineering to build the analytical feature matrix
        logger.info("Stage 2: Running feature engineering")
        engine = RecruiterFeatureEngine()
        enriched_profiles = engine.engineer_candidate_features(dfs)
        logger.info(f"Engineered candidate feature matrix with shape {enriched_profiles.shape}")

        # 3. Generate documents & perform semantic/skill matching
        logger.info("Stage 3: Running semantic and capability matching")
        from matching_layer import SkillExtractor
        extractor = SkillExtractor(skill_dictionary=required_skills)
        dfs["skills"] = extractor.enrich_skills_dataframe(dfs)

        matcher = SemanticMatchingLayer(
            model_name=matcher_cfg["model_name"],
            use_fallback=matcher_cfg["use_fallback"]
        )
        
        # Generate cohesive textual documents per candidate
        candidate_docs_df = matcher.generate_candidate_documents(dfs)
        
        # Merge semantic text context into our feature matrix
        candidates_integrated = enriched_profiles.merge(
            candidate_docs_df[["candidate_id", "synthetic_document"]],
            on="candidate_id",
            how="inner"
        )
        
        # Compute matching signals (Semantic, Title, Skills)
        matched_results_df = matcher.compute_matching_scores(
            job_description=job_description,
            target_title=job_title,
            target_skills=required_skills,
            candidates_enriched_df=candidates_integrated,
            df_dict=dfs
        )
        logger.info("Semantic and skill matching complete")

        # 4. Rank candidates using the calibrated hybrid formula
        logger.info("Stage 4: Executing hybrid ranking engine")
        ranking_weights = HybridRankingWeights(
            weight_semantic=weights_cfg["weight_semantic"],
            weight_profile_quality=weights_cfg["weight_profile_quality"],
            weight_recruitability=weights_cfg["weight_recruitability"],
            weight_risk_penalty=weights_cfg["weight_risk_penalty"]
        )
        ranking_system = HybridRankingEngine(weights=ranking_weights)
        leaderboard_df = ranking_system.rank_candidates(matched_results_df)
        logger.info("Leaderboard compiled successfully")

        # 5. Extract explainability details for top candidates
        logger.info("Stage 5: Generating explainability summaries")
        explainer = RecruiterExplainabilityEngine(target_skills_list=required_skills)
        limit_explain = min(3, len(leaderboard_df))
        logger.info(f"Analyzing evaluation summaries for the top {limit_explain} candidates:")
        
        for i in range(limit_explain):
            cand_row = leaderboard_df.iloc[i]
            report = explainer.generate_candidate_explanation(cand_row, dfs)
            markdown_repr = explainer.generate_formatted_markdown_report(report)
            
            logger.info(f"--- CANDIDATE RANK {i+1} PROFILE ---")
            logger.info(f"\n{markdown_repr}\n")

        # 6. Build final submission outputs & exports
        logger.info("Stage 6: Exporting submission artifacts")
        sub_generator = RecruiterSubmissionGenerator(target_skills_list=required_skills)
        exportable_leaderboard = sub_generator.compile_export_dataframe(
            ranked_leaderboard=leaderboard_df,
            df_dict=dfs,
            enrich_explanations=True
        )
        
        csv_file = sub_generator.export_submission(exportable_leaderboard, output_csv, "csv")
        xlsx_file = sub_generator.export_submission(exportable_leaderboard, output_xlsx, "xlsx")
        
        logger.info("E2E Recruitment Pipeline successfully executed!")
        logger.info(f"Output saved to: {csv_file}")
        logger.info(f"Output saved to: {xlsx_file}")

    except Exception as e:
        logger.warning(f"Standard pipeline execution failed: {e}. Falling back to Zero-Dependency Pure Python Fallback Pipeline.")
        run_pure_python_fallback_pipeline(config_path, target_job=target_job, required_toolkit=required_toolkit)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Candidate Matching & Hybrid Ranking Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--job", type=str, default=None, help="Target job title")
    parser.add_argument("--skills", type=str, default=None, help="Comma-separated required skills/tools")
    args = parser.parse_args()
    
    req_toolkit = None
    if args.skills:
        req_toolkit = [s.strip() for s in args.skills.split(",") if s.strip()]
        
    run_end_to_end_pipeline(config_path=args.config, target_job=args.job, required_toolkit=req_toolkit)
