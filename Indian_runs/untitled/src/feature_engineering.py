import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("FeatureEngineering")


class RecruiterFeatureEngine:
    """
    Feature engine specialized in deriving talent-focused features from relational tables,
    mapping candidates to standard variables like stability, prestige, responsiveness, and risks.
    """

    def __init__(self, baseline_date_str: str = "2026-06-20"):
        self.baseline_date = pd.to_datetime(baseline_date_str)
        logger.info(f"Initialized recruiter feature engine using baseline anchor date: {self.baseline_date.date()}")

    def engineer_candidate_features(self, df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Derives high-fidelity composite scores from disjoint databases.
        Combines:
            - Academic Prestige Score
            - Skill quality metrics
            - Stability and progression values
            - Dynamic platform interaction parameters
            - Notice, stale, job-hopping risk penalty indexes
            - Comprehensive quality score (objective baseline)
        """
        profiles_df = df_dict["profiles"].copy()
        careers_df = df_dict["careers"].copy()
        education_df = df_dict["education"].copy()
        skills_df = df_dict["skills"].copy()

        logger.info("Computing academic prestige index")
        academic_prestige = self._compute_academic_prestige(education_df)
        profiles_df = profiles_df.merge(academic_prestige, on="candidate_id", how="left")
        profiles_df["academic_prestige_score"] = profiles_df["academic_prestige_score"].fillna(0.2)

        logger.info("Computing skill profiles and endorsements")
        skill_metrics = self._compute_skill_quality_metrics(skills_df)
        profiles_df = profiles_df.merge(skill_metrics, on="candidate_id", how="left")
        profiles_df["avg_skill_endorsements"] = profiles_df["avg_skill_endorsements"].fillna(0.0)
        profiles_df["total_skills_count"] = profiles_df["total_skills_count"].fillna(0)

        logger.info("Analyzing career stability and tenures")
        career_stability = self._compute_career_stability(careers_df)
        profiles_df = profiles_df.merge(career_stability, on="candidate_id", how="left")
        profiles_df["career_stability_score"] = profiles_df["career_stability_score"].fillna(0.7)
        profiles_df["avg_role_duration_months"] = profiles_df["avg_role_duration_months"].fillna(0.0)
        profiles_df["num_companies_worked"] = profiles_df["num_companies_worked"].fillna(0)

        logger.info("Computing recruiter responsiveness signals")
        profiles_df["responsiveness_score"] = (
            profiles_df["recruiter_response_rate"] * 
            (1.0 - (profiles_df["avg_response_time_hours"].clip(upper=250.0) / 250.0))
        )
        profiles_df["interview_reliability"] = profiles_df["interview_completion_rate"]

        logger.info("Computing platform active engagement indicators")
        profiles_df["active_engagement_score"] = self._compute_active_engagement(profiles_df)

        profiles_df["verification_score"] = (
            profiles_df["verified_email"].astype(int) + 
            profiles_df["verified_phone"].astype(int) + 
            profiles_df["linkedin_connected"].astype(int)
        ) / 3.0

        assessment_cols = [col for col in profiles_df.columns if col.startswith("assessment_score_")]
        if assessment_cols:
            profiles_df["avg_assessment_score"] = profiles_df[assessment_cols].mean(axis=1) / 100.0
            profiles_df["avg_assessment_score"] = profiles_df["avg_assessment_score"].fillna(0.0)
        else:
            profiles_df["avg_assessment_score"] = 0.0

        profiles_df["platform_achievement_score"] = (
            (profiles_df["avg_assessment_score"] * 0.7) + 
            (profiles_df["github_activity_score"].clip(lower=0.0) / 100.0 * 0.3)
        ).clip(lower=0.0)

        expected_salary_mid = (profiles_df["salary_expected_min_lpa"] + profiles_df["salary_expected_max_lpa"]) / 2.0
        profiles_df["years_to_salary_ratio"] = (profiles_df["years_of_experience"] / expected_salary_mid.replace(0, np.nan)).fillna(0.3)
        profiles_df["salary_value_score"] = (profiles_df["years_to_salary_ratio"].clip(upper=1.5) / 1.5)

        logger.info("Evaluating candidate risk levels")
        profiles_df["notice_period_risk"] = profiles_df["notice_period_days"].apply(
            lambda x: 1.0 if x >= 90 else (0.5 if x >= 60 else 0.0)
        )
        
        days_since_active = (self.baseline_date - profiles_df["last_active_date"]).dt.days
        profiles_df["stale_profile_risk"] = (days_since_active.fillna(365).clip(lower=0, upper=180) / 180.0)
        profiles_df["job_hopping_risk"] = ((profiles_df["avg_role_duration_months"] < 12.0) & (profiles_df["num_companies_worked"] >= 2)).astype(float)

        profiles_df["aggregate_risk_score"] = (
            (profiles_df["notice_period_risk"] * 0.4) + 
            (profiles_df["stale_profile_risk"] * 0.3) + 
            (profiles_df["job_hopping_risk"] * 0.3)
        )

        profiles_df["profile_completeness_ratio"] = profiles_df["profile_completeness"] / 100.0
        
        # Override and completely disable secondary modifiers as requested (availability, immediate, collab, stability)
        profiles_df["career_stability_score"] = 1.0
        profiles_df["responsiveness_score"] = 1.0
        profiles_df["active_engagement_score"] = 1.0
        profiles_df["notice_period_risk"] = 0.0
        profiles_df["stale_profile_risk"] = 0.0
        profiles_df["job_hopping_risk"] = 0.0
        profiles_df["aggregate_risk_score"] = 0.0
        
        profiles_df["composite_quality_score"] = (
            (profiles_df["academic_prestige_score"] * 0.20) +
            (profiles_df["career_stability_score"] * 0.25) +
            (profiles_df["platform_achievement_score"] * 0.20) +
            (profiles_df["verification_score"] * 0.15) +
            (profiles_df["salary_value_score"] * 0.10) +
            (profiles_df["profile_completeness_ratio"] * 0.10)
        )

        logger.info(f"Composite candidate feature tables merged. Grid dimensions: {profiles_df.shape}")
        return profiles_df

    def _compute_academic_prestige(self, education_df: pd.DataFrame) -> pd.DataFrame:
        """
        Translates educational pedigree prestige and degree levels into scholastic scores.
        """
        if education_df.empty:
            return pd.DataFrame(columns=["candidate_id", "academic_prestige_score"])

        tier_weights = {
            "tier_1": 1.0,
            "tier_2": 0.75,
            "tier_3": 0.50,
            "tier_4": 0.30,
            "unknown": 0.20
        }

        education_df = education_df.copy()
        education_df["tier_score"] = education_df["prestige_tier"].map(tier_weights).fillna(0.2)
        
        def get_degree_multiplier(degree: str) -> float:
            if not isinstance(degree, str):
                return 0.8
            d_lower = degree.lower()
            if "ph.d" in d_lower or "phd" in d_lower:
                return 1.25
            if "m.tech" in d_lower or "m.s" in d_lower or "m.sc" in d_lower or "m.e" in d_lower or "mba" in d_lower:
                return 1.15
            if "b.tech" in d_lower or "b.e" in d_lower or "b.sc" in d_lower or "b.a" in d_lower:
                return 1.0
            return 0.85

        education_df["degree_multiplier"] = education_df["degree"].apply(get_degree_multiplier)
        education_df["weighted_edu_score"] = education_df["tier_score"] * education_df["degree_multiplier"]

        academic_score = education_df.groupby("candidate_id")["weighted_edu_score"].max().reset_index()
        academic_score.columns = ["candidate_id", "academic_prestige_score"]
        academic_score["academic_prestige_score"] = academic_score["academic_prestige_score"].clip(upper=1.0)
        return academic_score

    def _compute_skill_quality_metrics(self, skills_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exposes skills abundance and logarithmic-decayed endorsements.
        """
        if skills_df.empty:
            return pd.DataFrame(columns=["candidate_id", "avg_skill_endorsements", "total_skills_count"])

        skill_agg = skills_df.groupby("candidate_id").agg(
            avg_skill_endorsements=("endorsements_count", "mean"),
            total_skills_count=("skill_name", "count")
        ).reset_index()

        skill_agg["avg_skill_endorsements"] = np.log1p(skill_agg["avg_skill_endorsements"]) / np.log1p(100.0)
        skill_agg["avg_skill_endorsements"] = skill_agg["avg_skill_endorsements"].clip(upper=1.0)
        return skill_agg

    def _compute_career_stability(self, careers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Determines tenure progression, average employment timeline lengths, and promotion signals.
        """
        if careers_df.empty:
            return pd.DataFrame(columns=["candidate_id", "career_stability_score", "avg_role_duration_months", "num_companies_worked"])

        careers_df = careers_df.copy()
        career_agg = careers_df.groupby("candidate_id").agg(
            total_duration_months=("duration_months", "sum"),
            avg_role_duration_months=("duration_months", "mean"),
            num_companies_worked=("company", "nunique"),
            num_positions=("title", "count")
        ).reset_index()

        career_agg["career_stability_score"] = (career_agg["avg_role_duration_months"] / 36.0).clip(lower=0.2, upper=1.0)

        churn_mask = (career_agg["num_companies_worked"] >= 3) & (career_agg["avg_role_duration_months"] < 14.0)
        career_agg.loc[churn_mask, "career_stability_score"] *= 0.65

        company_loyalty_agg = careers_df.groupby(["candidate_id", "company"])["title"].count().reset_index()
        promoted_candidates = company_loyalty_agg[company_loyalty_agg["title"] > 1]["candidate_id"].unique()
        
        career_agg.loc[career_agg["candidate_id"].isin(promoted_candidates), "career_stability_score"] *= 1.15
        career_agg["career_stability_score"] = career_agg["career_stability_score"].clip(upper=1.0)

        return career_agg[["candidate_id", "career_stability_score", "avg_role_duration_months", "num_companies_worked"]]

    def _compute_active_engagement(self, profiles_df: pd.DataFrame) -> pd.Series:
        """
        Evaluates active engagement of candidates using dynamic actions and logins.
        """
        days_gap = (self.baseline_date - profiles_df["last_active_date"]).dt.days
        days_gap = days_gap.fillna(180)
        
        time_decay = np.exp(-days_gap / 45.0)

        search_vol = profiles_df["search_appearance_30d"].clip(upper=500) / 500.0
        saved_vol = profiles_df["saved_by_recruiters_30d"].clip(upper=15) / 15.0
        view_vol = profiles_df["profile_views_30d"].clip(upper=100) / 100.0

        aei_raw = (
            (time_decay * 0.45) + 
            (search_vol * 0.20) + 
            (view_vol * 0.15) + 
            (saved_vol * 0.10) +
            (profiles_df["open_to_work"].astype(float) * 0.10)
        )
        return aei_raw.clip(lower=0.0, upper=1.0)


if __name__ == "__main__":
    from data_loader import CandidateDataLoader
    
    proj_root = Path(__file__).parent.parent
    test_schema = proj_root / "data" / "candidate_schema.json"
    test_candidates = proj_root / "data" / "sample_candidates.json"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Verifying RecruiterFeatureEngine execution...")
    
    try:
        loader = CandidateDataLoader(schema_path=test_schema)
        raw_list = loader.load_candidates_raw(test_candidates)
        dfs = loader.parse_to_relational_dataframes(raw_list)
        
        engine = RecruiterFeatureEngine()
        enriched_profiles = engine.engineer_candidate_features(dfs)
        logger.info("Feature engineering module successfully verified")
    except Exception as e:
        logger.error(f"Verification check failed: {e}", exc_info=True)
        sys.exit(1)
