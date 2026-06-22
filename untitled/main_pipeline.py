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

try:
    from data_loader import CandidateDataLoader
    from feature_engineering import RecruiterFeatureEngine
    from matching_layer import SemanticMatchingLayer
    from hybrid_ranking_engine import HybridRankingWeights, HybridRankingEngine
    from explainability_pipeline import RecruiterExplainabilityEngine
    from submission_generator import RecruiterSubmissionGenerator
except ImportError as e:
    logger.critical(f"Failed to resolve modular package imports under src/ path: {e}")
    sys.exit(1)


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


def run_end_to_end_pipeline(config_path: str = "config.json") -> None:
    """
    Executes the entire end-to-end recruitment prioritization and matching flow.
    """
    logger.info("Initializing Candidate Matching & Hybrid Ranking Pipeline")

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

    # 1. Load profiles and relational tables
    logger.info("Stage 1: Loading candidate profiles")
    try:
        loader = CandidateDataLoader(schema_path=schema_path)
        raw_candidates_list = loader.load_candidates_raw(candidates_path)
        dfs = loader.parse_to_relational_dataframes(raw_candidates_list)
        for df_name, df_obj in dfs.items():
            logger.info(f"Table '{df_name}': {df_obj.shape[0]} records loaded")
    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Run Feature Engineering to build the analytical feature matrix
    logger.info("Stage 2: Running feature engineering")
    try:
        engine = RecruiterFeatureEngine()
        enriched_profiles = engine.engineer_candidate_features(dfs)
        logger.info(f"Engineered candidate feature matrix with shape {enriched_profiles.shape}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Generate documents & perform semantic/skill matching
    logger.info("Stage 3: Running semantic and capability matching")
    try:
        from matching_layer import SkillExtractor
        extractor = SkillExtractor(skill_dictionary=job_cfg["required_skills"])
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
            job_description=job_cfg["job_description"],
            target_title=job_cfg["job_title"],
            target_skills=job_cfg["required_skills"],
            candidates_enriched_df=candidates_integrated,
            df_dict=dfs
        )
        logger.info("Semantic and skill matching complete")
    except Exception as e:
        logger.error(f"Semantic matching failed: {e}", exc_info=True)
        sys.exit(1)

    # 4. Rank candidates using the calibrated hybrid formula
    logger.info("Stage 4: Executing hybrid ranking engine")
    try:
        ranking_weights = HybridRankingWeights(
            weight_semantic=weights_cfg["weight_semantic"],
            weight_profile_quality=weights_cfg["weight_profile_quality"],
            weight_recruitability=weights_cfg["weight_recruitability"],
            weight_risk_penalty=weights_cfg["weight_risk_penalty"]
        )
        ranking_system = HybridRankingEngine(weights=ranking_weights)
        leaderboard_df = ranking_system.rank_candidates(matched_results_df)
        logger.info("Leaderboard compiled successfully")
    except Exception as e:
        logger.error(f"Ranking computation failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Extract explainability details for top candidates
    logger.info("Stage 5: Generating explainability summaries")
    try:
        explainer = RecruiterExplainabilityEngine(target_skills_list=job_cfg["required_skills"])
        limit_explain = min(3, len(leaderboard_df))
        logger.info(f"Analyzing evaluation summaries for the top {limit_explain} candidates:")
        
        for i in range(limit_explain):
            cand_row = leaderboard_df.iloc[i]
            report = explainer.generate_candidate_explanation(cand_row, dfs)
            markdown_repr = explainer.generate_formatted_markdown_report(report)
            
            logger.info(f"--- CANDIDATE RANK {i+1} PROFILE ---")
            logger.info(f"\n{markdown_repr}\n")
    except Exception as e:
        logger.error(f"Explainability generation failed: {e}", exc_info=True)
        sys.exit(1)

    # 6. Build final submission outputs & exports
    logger.info("Stage 6: Exporting submission artifacts")
    try:
        sub_generator = RecruiterSubmissionGenerator(target_skills_list=job_cfg["required_skills"])
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
        logger.error(f"Submission generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_end_to_end_pipeline()
