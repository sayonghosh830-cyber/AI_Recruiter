import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import jsonschema
from jsonschema import validate

logger = logging.getLogger("DataLoader")


class CandidateDataLoader:
    """
    DataLoader responsible for reading, schema-validating, and structured-parses
    nested candidate JSON records into normal/relational Pandas DataFrames.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema: Optional[Dict[str, Any]] = None
        if schema_path:
            self.schema = self.load_schema(schema_path)

    @staticmethod
    def load_schema(schema_path: str | Path) -> Dict[str, Any]:
        """
        Loads the json validation schema from disk.
        """
        path = Path(schema_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON validation schema not found at: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        logger.info(f"Loaded schema configuration from {path.name}")
        return schema

    @staticmethod
    def load_candidates_raw(file_path: str | Path) -> List[Dict[str, Any]]:
        """
        Loads the candidate profiles raw JSON list.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Candidate JSON file not found at: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        logger.info(f"Loaded {len(candidates)} candidate profiles from raw source")
        return candidates

    def validate_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates candidate JSON dictionary schema against the configured ruleset.
        """
        if not self.schema:
            return True, None
            
        try:
            validate(instance=candidate, schema=self.schema)
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            error_msg = f"Validation failed for candidate {candidate.get('candidate_id', 'UNKNOWN')}: {e.message}"
            return False, error_msg

    def parse_to_relational_dataframes(self, raw_candidates: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Normalizes nested structures into clean relational structures.
        Returns:
            Dict containing:
                'profiles': Base demographics and redrob platform signal metrics.
                'careers': Job history/experience details.
                'education': Academic milestones and certification list.
                'skills': Skills inventory and duration details.
        """
        profiles_list = []
        careers_list = []
        education_list = []
        skills_list = []
        
        valid_count = 0
        invalid_count = 0
        
        for cand in raw_candidates:
            is_valid, validation_err = self.validate_candidate(cand)
            if not is_valid:
                logger.warning(validation_err)
                invalid_count += 1
                continue
            
            valid_count += 1
            cand_id = cand["candidate_id"]
            
            # Demographic & platform details
            prof = cand.get("profile", {})
            sig = cand.get("redrob_signals", {})
            
            profile_data = {
                "candidate_id": cand_id,
                "name": prof.get("anonymized_name"),
                "headline": prof.get("headline"),
                "summary": prof.get("summary"),
                "location": prof.get("location"),
                "country": prof.get("country"),
                "years_of_experience": float(prof.get("years_of_experience", 0)),
                "current_title": prof.get("current_title"),
                "current_company": prof.get("current_company"),
                "current_company_size": prof.get("current_company_size"),
                "current_industry": prof.get("current_industry"),
                
                # Platform dynamics and responsiveness weights
                "profile_completeness": float(sig.get("profile_completeness_score", 0)),
                "signup_date": sig.get("signup_date"),
                "last_active_date": sig.get("last_active_date"),
                "open_to_work": bool(sig.get("open_to_work_flag", False)),
                "profile_views_30d": int(sig.get("profile_views_received_30d", 0)),
                "applications_submitted_30d": int(sig.get("applications_submitted_30d", 0)),
                "recruiter_response_rate": float(sig.get("recruiter_response_rate", 0.0)),
                "avg_response_time_hours": float(sig.get("avg_response_time_hours", 0.0)),
                "connection_count": int(sig.get("connection_count", 0)),
                "endorsements_received": int(sig.get("endorsements_received", 0)),
                "notice_period_days": int(sig.get("notice_period_days", 0)),
                "salary_expected_min_lpa": float(sig.get("expected_salary_range_inr_lpa", {}).get("min", 0.0)),
                "salary_expected_max_lpa": float(sig.get("expected_salary_range_inr_lpa", {}).get("max", 0.0)),
                "preferred_work_mode": sig.get("preferred_work_mode"),
                "willing_to_relocate": bool(sig.get("willing_to_relocate", False)),
                "github_activity_score": float(sig.get("github_activity_score", -1.0)),
                "search_appearance_30d": int(sig.get("search_appearance_30d", 0)),
                "saved_by_recruiters_30d": int(sig.get("saved_by_recruiters_30d", 0)),
                "interview_completion_rate": float(sig.get("interview_completion_rate", 0.0)),
                "offer_acceptance_rate": float(sig.get("offer_acceptance_rate", -1.0)),
                "verified_email": bool(sig.get("verified_email", False)),
                "verified_phone": bool(sig.get("verified_phone", False)),
                "linkedin_connected": bool(sig.get("linkedin_connected", False))
            }
            
            # Map specific verified assessment indicators as standard parameters if defined
            skill_assessments = sig.get("skill_assessment_scores", {})
            for skill_assessment_name, score in skill_assessments.items():
                profile_data[f"assessment_score_{skill_assessment_name.lower().replace(' ', '_')}"] = float(score)
            
            profiles_list.append(profile_data)
            
            # Extract historical career tables
            for exp in cand.get("career_history", []):
                career_data = {
                    "candidate_id": cand_id,
                    "company": exp.get("company"),
                    "title": exp.get("title"),
                    "start_date": exp.get("start_date"),
                    "end_date": exp.get("end_date"),
                    "duration_months": int(exp.get("duration_months", 0)),
                    "is_current": bool(exp.get("is_current", False)),
                    "industry": exp.get("industry"),
                    "company_size": exp.get("company_size"),
                    "description": exp.get("description", "")
                }
                careers_list.append(career_data)
                
            # Extract scholarly details
            for edu in cand.get("education", []):
                edu_data = {
                    "candidate_id": cand_id,
                    "institution": edu.get("institution"),
                    "degree": edu.get("degree"),
                    "field_of_study": edu.get("field_of_study"),
                    "start_year": int(edu.get("start_year", 0)) if edu.get("start_year") else None,
                    "end_year": int(edu.get("end_year", 0)) if edu.get("end_year") else None,
                    "grade": edu.get("grade"),
                    "prestige_tier": edu.get("tier", "unknown")
                }
                education_list.append(edu_data)
                
            # Extract tools/skills database
            for skill in cand.get("skills", []):
                skill_data = {
                    "candidate_id": cand_id,
                    "skill_name": skill.get("name"),
                    "proficiency": skill.get("proficiency"),
                    "endorsements_count": int(skill.get("endorsements", 0)),
                    "usage_months": int(skill.get("duration_months", 0)) if skill.get("duration_months") else None
                }
                skills_list.append(skill_data)
                
        # Parse output dataframes
        profiles_df = pd.DataFrame(profiles_list)
        careers_df = pd.DataFrame(careers_list)
        education_df = pd.DataFrame(education_list)
        skills_df = pd.DataFrame(skills_list)
        
        # Datetime casting for analysis
        for col in ["signup_date", "last_active_date"]:
            if col in profiles_df.columns:
                profiles_df[col] = pd.to_datetime(profiles_df[col], errors="coerce")
                
        for col in ["start_date", "end_date"]:
            if col in careers_df.columns:
                careers_df[col] = pd.to_datetime(careers_df[col], errors="coerce")

        logger.info(
            f"Relational normalization complete. "
            f"Profiles: {profiles_df.shape[0]}, "
            f"Careers: {careers_df.shape[0]}, "
            f"Education: {education_df.shape[0]}, "
            f"Skills: {skills_df.shape[0]}"
        )
        logger.info(f"Loaded records validation checklist: Valid = {valid_count}, Invalid/Skipped = {invalid_count}")
        
        return {
            "profiles": profiles_df,
            "careers": careers_df,
            "education": education_df,
            "skills": skills_df
        }

    @staticmethod
    def profile_dataset_statistics(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Performs high-level data profiling diagnostics to expose aggregate characteristics
        of the talent pool. Contains zero fluff.
        """
        profiles = df_dict["profiles"]
        careers = df_dict["careers"]
        skills = df_dict["skills"]
        education = df_dict["education"]
        
        stats = {}
        
        stats["candidate_count"] = int(profiles.shape[0])
        stats["profile_completeness_avg"] = float(profiles["profile_completeness"].mean())
        
        open_to_work_counts = profiles["open_to_work"].value_counts(normalize=True)
        stats["open_to_work_ratio"] = float(open_to_work_counts.get(True, 0.0))
        
        linked_github_count = (profiles["github_activity_score"] >= 0).sum()
        stats["github_linked_ratio"] = float(linked_github_count / len(profiles))
        
        if linked_github_count > 0:
            stats["github_active_score_avg"] = float(profiles.loc[profiles["github_activity_score"] >= 0, "github_activity_score"].mean())
        else:
            stats["github_active_score_avg"] = 0.0
        
        stats["avg_response_rate_pct"] = float(profiles["recruiter_response_rate"].mean() * 100)
        stats["avg_response_time_hrs"] = float(profiles["avg_response_time_hours"].mean())
        stats["avg_notice_period_days"] = float(profiles["notice_period_days"].mean())
        stats["salary_expected_min_lpa_avg"] = float(profiles["salary_expected_min_lpa"].mean())
        stats["salary_expected_max_lpa_avg"] = float(profiles["salary_expected_max_lpa"].mean())
        
        avg_roles_per_candidate = len(careers) / len(profiles) if len(profiles) > 0 else 0
        avg_skills_per_candidate = len(skills) / len(profiles) if len(profiles) > 0 else 0
        stats["avg_roles_held"] = float(avg_roles_per_candidate)
        stats["avg_skills_listed"] = float(avg_skills_per_candidate)
        
        if "prestige_tier" in education.columns:
            tier_pcts = education["prestige_tier"].value_counts(normalize=True).to_dict()
            stats["academic_tier_distribution"] = {str(k): float(v) for k, v in tier_pcts.items()}
            
        logger.info(f"Dataset summary diagnostics completed across {stats['candidate_count']} candidate profiles.")
        return stats


if __name__ == "__main__":
    proj_root = Path(__file__).parent.parent
    test_schema = proj_root / "data" / "candidate_schema.json"
    test_candidates = proj_root / "data" / "sample_candidates.json"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Verifying CandidateDataLoader module execution...")
    
    try:
        loader = CandidateDataLoader(schema_path=test_schema)
        raw_list = loader.load_candidates_raw(test_candidates)
        dfs = loader.parse_to_relational_dataframes(raw_list)
        report = loader.profile_dataset_statistics(dfs)
        logger.info("Module verification succeeded")
    except Exception as e:
        logger.error(f"Loader execution verification failed: {e}", exc_info=True)
        sys.exit(1)
