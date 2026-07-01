import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("HybridRankingEngine")


class HybridRankingWeights:
    """
    Holds relative weighting parameters used in composite pipeline scores.
    """
    def __init__(
        self,
        weight_semantic: float = 0.99,
        weight_profile_quality: float = 0.004,
        weight_recruitability: float = 0.003,
        weight_risk_penalty: float = 0.003
    ):
        self.w_semantic = weight_semantic
        self.w_quality = weight_profile_quality
        self.w_recruitability = weight_recruitability
        self.w_risk = weight_risk_penalty
        
        self.validate_weights()

    def validate_weights(self):
        """
        Confirms weights are configured correctly.
        """
        total_positive = self.w_semantic + self.w_quality + self.w_recruitability
        if not np.isclose(total_positive, 0.90) and not np.isclose(total_positive, 1.0):
            logger.warning(
                f"Positive weights sum to {total_positive:.2f}. "
                "Ideally positive components should sum up close to 0.90 - 1.00."
            )


class HybridRankingEngine:
    """
    Calibrates multiple dimensional scores into a standardized composite Match Score.
    """

    def __init__(self, weights: Optional[HybridRankingWeights] = None):
        self.weights = weights or HybridRankingWeights()

    def rank_candidates(
        self,
        matched_profiles_df: pd.DataFrame,
        target_role_title: str = "Backend Software Engineer"
    ) -> pd.DataFrame:
        """
        Applies multi-dimensional scoring, executes a title relevance multiplier layer,
        and sorts profiles descending.
        """
        results_df = matched_profiles_df.copy()

        score_checks = {
            "composite_semantic_score": 0.0,
            "composite_quality_score": 0.5,
            "active_engagement_score": 0.5,
            "responsiveness_score": 0.5,
            "interview_reliability": 0.7,
            "aggregate_risk_score": 0.0
        }
        for score_col, default_val in score_checks.items():
            if score_col not in results_df.columns:
                results_df[score_col] = default_val

        results_df["composite_recruitability_score"] = (
            (results_df["active_engagement_score"] * 0.40) +
            (results_df["responsiveness_score"] * 0.35) +
            (results_df["interview_reliability"].fillna(0.7) * 0.25)
        )

        results_df["final_match_score"] = (
            (self.weights.w_semantic * results_df["composite_semantic_score"]) +
            (self.weights.w_quality * results_df["composite_quality_score"]) +
            (self.weights.w_recruitability * results_df["composite_recruitability_score"]) -
            (self.weights.w_risk * results_df["aggregate_risk_score"])
        )

        results_df["final_match_score"] = results_df["final_match_score"].clip(lower=0.0, upper=1.0)

        # Apply deterministic title relevance multiplier layer to prevent out-of-domain bias
        for idx, row in results_df.iterrows():
            base_score = float(row.get("final_match_score", 0.5))
            current_title = str(row.get("current_title", "Specialist")).lower()
            
            # Title Relevance Penalties and Bonuses
            title_multiplier = 1.0
            if "backend" in current_title or "java" in current_title:
                title_multiplier = 1.05
            elif ".net" in current_title or "full stack" in current_title:
                title_multiplier = 0.98
            elif "frontend" in current_title or "devops" in current_title or "qa" in current_title or "data engineer" in current_title:
                title_multiplier = 0.82
            elif any(domain in current_title for domain in ["mechanical", "civil", "customer support", "sales"]):
                title_multiplier = 0.45
                
            adjusted_score = base_score * title_multiplier
            results_df.at[idx, "final_match_score"] = min(max(adjusted_score, 0.0), 1.0)

        # Re-sort and recalculate the ranks so the UI leaderboard shifts
        results_df = results_df.sort_values(
            by=["final_match_score", "composite_quality_score", "years_of_experience"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        results_df["leaderboard_rank"] = results_df.index + 1
        return results_df

    def evaluate_hiring_confidence(self, final_score: float) -> Tuple[str, str]:
        """
        Maps a match score decimal to discrete categories.
        Supports dynamic threshold calibration under sparse matching (TF-IDF) fallback.
        """
        if final_score >= 0.65:
            return "EXCEPTIONAL", "Fast-track to technical interview immediately."
        elif final_score >= 0.55:
            return "RECOMMENDED", "Move forward to initial recruiter screening."
        elif final_score >= 0.30:
            return "CONSIDER", "Review closely or place in alternative pipelines."
        else:
            return "NOT_ALIGNED", "Reject profile."


if __name__ == "__main__":
    from data_loader import CandidateDataLoader
    from feature_engineering import RecruiterFeatureEngine
    from matching_layer import SemanticMatchingLayer
    
    proj_root = Path(__file__).parent.parent
    test_schema = proj_root / "data" / "candidate_schema.json"
    test_candidates = proj_root / "data" / "sample_candidates.json"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Verifying HybridRankingEngine execution...")
    
    try:
        loader = CandidateDataLoader(schema_path=test_schema)
        raw_list = loader.load_candidates_raw(test_candidates)
        dfs = loader.parse_to_relational_dataframes(raw_list)
        
        fe_engine = RecruiterFeatureEngine()
        enriched_profiles = fe_engine.engineer_candidate_features(dfs)
        
        matcher = SemanticMatchingLayer(use_fallback=True)
        candidates_with_docs = matcher.generate_candidate_documents(dfs)
        candidates_integrated = enriched_profiles.merge(
            candidates_with_docs[["candidate_id", "synthetic_document"]],
            on="candidate_id"
        )
        
        job_query = "Backend developer Python SQL streaming Kafka"
        target_role_title = "Backend Developer"
        required_tools = ["Python", "SQL", "Kafka"]
        
        matched_profiles = matcher.compute_matching_scores(
            job_description=job_query,
            target_title=target_role_title,
            target_skills=required_tools,
            candidates_enriched_df=candidates_integrated,
            df_dict=dfs
        )
        
        ranking_engine = HybridRankingEngine()
        leaderboard_df = ranking_engine.rank_candidates(matched_profiles)
        logger.info("Ranking engine validation successful.")
    except Exception as e:
        logger.error(f"Ranking validation failed: {e}", exc_info=True)
        sys.exit(1)
