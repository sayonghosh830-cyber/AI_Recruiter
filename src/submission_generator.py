import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd

from explainability_pipeline import RecruiterExplainabilityEngine
from hybrid_ranking_engine import HybridRankingEngine

logger = logging.getLogger("SubmissionGenerator")


class SubmissionValidationException(ValueError):
    """Raised when generated rank arrays violate consistency or boundary safety gates."""
    pass


class RecruiterSubmissionGenerator:
    """
    Validates, serializes, and saves the final prioritised leaderboard to filesystem datasets.
    """

    def __init__(self, target_skills_list: Optional[List[str]] = None):
        self.target_skills = target_skills_list or []
        self.ranking_engine = HybridRankingEngine()
        self.explainer = RecruiterExplainabilityEngine(target_skills_list=self.target_skills)

    def validate_leaderboard(self, df: pd.DataFrame) -> None:
        """
        Executes strict verification bounds to ensure safety and structural integrity of the output.
        """
        if df.empty:
            raise SubmissionValidationException("Hiring leaderboard is empty. Cannot continue export.")

        required_cols = ["candidate_id", "leaderboard_rank", "final_match_score"]
        for col in required_cols:
            if col not in df.columns:
                raise SubmissionValidationException(f"Missing required leaderboard column: {col}")

        # Unique candidates verification
        duplicates = df["candidate_id"].duplicated().sum()
        if duplicates > 0:
            raise SubmissionValidationException(f"Candidate records contain {duplicates} duplicate candidate IDs")

        # Rank integrity and sequence checks
        ranks = df["leaderboard_rank"].values
        sorted_ranks = np.sort(ranks)
        
        if sorted_ranks[0] != 1:
            raise SubmissionValidationException(f"Rank sequence must start at 1, found starting value of {sorted_ranks[0]}")
            
        diffs = np.diff(sorted_ranks)
        if not np.all(diffs == 1):
            raise SubmissionValidationException("Ranks are non-contiguous. Ranks must be monotonic continuous integers.")

        # Match score sort validation (Strictly decreasing matching confidence)
        scores = df["final_match_score"].values
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                raise SubmissionValidationException(
                    f"Lead sorting constraint broken. Score {scores[i]:.4f} at rank {i+1} is lower "
                    f"than target candidate score {scores[i+1]:.4f} at rank {i+2}"
                )

        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            raise SubmissionValidationException("Computed Match scores contain invalid NaN or Infinite values.")

        if np.any(scores < 0.0) or np.any(scores > 1.05):
            raise SubmissionValidationException("Matched scoring metrics breached baseline [0, 1] bounds.")

        logger.info("Leaderboard safety validation complete.")

    def compile_export_dataframe(
        self,
        ranked_leaderboard: pd.DataFrame,
        df_dict: Dict[str, pd.DataFrame],
        enrich_explanations: bool = True
    ) -> pd.DataFrame:
        """
        Builds a flattened, recruiter-friendly DataFrame containing scores, recommendations, and parameters.
        """
        self.validate_leaderboard(ranked_leaderboard)
        
        export_records = []
        for _, row in ranked_leaderboard.iterrows():
            cand_id = row["candidate_id"]
            name = row.get("name", "Anonymized Candidate")
            rank = int(row["leaderboard_rank"])
            score = float(row["final_match_score"])
            
            confidence_level, _ = self.ranking_engine.evaluate_hiring_confidence(score)
            
            explanation_data = self.explainer.generate_candidate_explanation(row, df_dict)
            reasoning = explanation_data["one_liner_reasoning"]
            action = explanation_data["hiring_action_recommendation"]
            
            record = {
                "candidate_id": cand_id,
                "name": name,
                "leaderboard_rank": rank,
                "final_match_score": round(score, 4),
                "hiring_confidence": confidence_level,
                "one_liner_reasoning": reasoning,
                "suggested_action": action,
                "matched_skills": ", ".join(explanation_data["matched_skills"]),
                "missing_skills": ", ".join(explanation_data["missing_skills"])
            }
            
            if enrich_explanations:
                record["stability_score"] = round(float(row.get("career_stability_score", 0.5)), 2)
                record["academic_prestige"] = round(float(row.get("academic_prestige_score", 0.0)), 2)
                record["platform_responsiveness"] = round(float(row.get("recruiter_response_rate", 0.0)), 2)
                record["attendance_reliability"] = round(float(row.get("interview_reliability", 0.7)), 2)
                record["churn_risk"] = round(float(row.get("job_hopping_risk", 0.0)), 2)
                record["notice_period_days"] = int(row.get("notice_period_days", 0))

            export_records.append(record)

        return pd.DataFrame(export_records)

    def export_submission(
        self,
        export_df: pd.DataFrame,
        output_path: Union[str, Path],
        file_format: str = "csv"
    ) -> Path:
        """
        Writes export frame to target local directory standard paths.
        Also automatically generates a platform-ready submission file 'data/submission.csv'.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        fmt = file_format.strip().lower()
        if fmt == "csv":
            export_df.to_csv(out_p, index=False)
            logger.info(f"Exported data saved directly to {out_p}")
        elif fmt == "xlsx":
            try:
                export_df.to_excel(out_p, index=False, engine="openpyxl")
                logger.info(f"Exported data saved directly to {out_p}")
            except Exception as e:
                logger.warning(f"Excel writer failed: {e}. Falling back to CSV.")
                fallback_path = out_p.with_suffix(".csv")
                export_df.to_csv(fallback_path, index=False)
                out_p = fallback_path
        else:
            raise ValueError(f"Unknown target file format flag specified: {file_format}")
            
        # Automatically export a strictly formatted Redrob Platform compliant submission file
        try:
            submission_df = pd.DataFrame()
            submission_df["candidate_id"] = export_df["candidate_id"]
            submission_df["rank"] = export_df["leaderboard_rank"]
            submission_df["score"] = export_df["final_match_score"]
            submission_df["reasoning"] = export_df["one_liner_reasoning"]
            
            sub_file_path = out_p.parent / "submission.csv"
            submission_df.to_csv(sub_file_path, index=False)
            logger.info(f"Platform-ready submission file automatically compiled at {sub_file_path}")
        except Exception as e:
            logger.warning(f"Could not build standard platform-ready submission file: {e}")
            
        return out_p


if __name__ == "__main__":
    from data_loader import CandidateDataLoader
    from feature_engineering import RecruiterFeatureEngine
    from matching_layer import SemanticMatchingLayer
    
    proj_root = Path(__file__).parent.parent
    test_schema = proj_root / "data" / "candidate_schema.json"
    test_candidates = proj_root / "data" / "sample_candidates.json"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Verifying RecruiterSubmissionGenerator module execution...")

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
        
        generator = RecruiterSubmissionGenerator(target_skills_list=required_tools)
        generator.validate_leaderboard(leaderboard_df)
        csv_export_df = generator.compile_export_dataframe(leaderboard_df, dfs, enrich_explanations=True)
        
        target_csv_file = proj_root / "data" / "leaderboard_preview.csv"
        generator.export_submission(csv_export_df, target_csv_file, "csv")
        
        if target_csv_file.exists():
            target_csv_file.unlink()
            
        logger.info("Submission generator module successfully verified.")
    except Exception as e:
        logger.error(f"Verification check failed: {e}", exc_info=True)
        sys.exit(1)
