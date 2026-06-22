import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ExplainabilityPipeline")


class RecruiterExplainabilityEngine:
    """
    Translates raw feature scores and statistical indicators into human-readable evaluations
    and recommendations for hiring managers.
    """

    def __init__(self, target_skills_list: Optional[List[str]] = None):
        self.target_skills = target_skills_list or []

    def generate_candidate_explanation(
        self,
        candidate_row: pd.Series,
        df_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Parses multi-dimensional profile scores to create a detailed qualitative evaluation report.
        """
        cand_id = candidate_row["candidate_id"]
        name = candidate_row.get("name", "Anonymized Candidate")
        current_title = candidate_row.get("current_title", "Specialist")
        final_score = float(candidate_row["final_match_score"])
        
        matched_skills = []
        missing_skills = []
        
        if self.target_skills:
            skills_df = df_dict["skills"]
            cand_skills_rows = skills_df[skills_df["candidate_id"] == cand_id]
            cand_skills_lower = [str(s).strip().lower() for s in cand_skills_rows["skill_name"].tolist() if pd.notna(s)]
            cand_skills_orig = {str(s).strip().lower(): str(s).strip() for s in cand_skills_rows["skill_name"].tolist() if pd.notna(s)}
            
            for req_skill in self.target_skills:
                req_norm = req_skill.strip().lower()
                found = False
                for cand_skill in cand_skills_lower:
                    if req_norm == cand_skill or req_norm in cand_skill or cand_skill in req_norm:
                        matched_skills.append(cand_skills_orig.get(cand_skill, req_skill))
                        found = True
                        break
                if not found:
                    missing_skills.append(req_skill)

        strengths = []
        semantic_sim = float(candidate_row.get("semantic_similarity_score", 0.0))
        title_match = float(candidate_row.get("title_match_score", 0.0))
        skill_overlap = float(candidate_row.get("skill_overlap_score", 0.0))
        
        if semantic_sim >= 0.70:
            strengths.append(f"Strong overall professional alignment with core resume requirements (Match score: {semantic_sim*100:.1f}%).")
        if title_match >= 0.70:
            strengths.append(f"Direct alignment with target job title (previously operated as '{current_title}').")
        if skill_overlap >= 0.75:
            strengths.append(f"High technical coverage with strong hard-skill overlaps ({len(matched_skills)} skills matched).")
        elif len(matched_skills) >= 2:
            strengths.append(f"Demonstrated proficiency in core tools: {', '.join(matched_skills[:3])}.")

        stability = float(candidate_row.get("career_stability_score", 0.5))
        prestige = float(candidate_row.get("academic_prestige_score", 0.2))
        
        if stability >= 0.85:
            strengths.append("High tenure stability with consistent organizational longevity.")
        elif stability >= 0.70:
            strengths.append("Consistent career growth and progression across organizations.")
        if prestige >= 0.75:
            strengths.append("High-quality educational pedigree from recognized tier-1/tier-2 academic institutions.")

        response_rate = float(candidate_row.get("recruiter_response_rate", 0.0))
        response_hrs = float(candidate_row.get("avg_response_time_hours", 200.0))
        reliability = float(candidate_row.get("interview_reliability", 0.5))
        github_val = float(candidate_row.get("github_activity_score", -1.0))
        
        if response_rate >= 0.75:
            strengths.append(f"Highly responsive collaborator with a partner response rate of {response_rate*100:.1f}%.")
        if response_hrs <= 24.0:
            strengths.append(f"Rapid recruiter response cycle time ({response_hrs:.1f} hours average).")
        if reliability >= 0.80:
            strengths.append(f"High interview completion reliability ({reliability*100:.1f}%).")
        if github_val >= 25.0:
            strengths.append(f"Visible open-source presence (Verified GitHub activity: {github_val:.1f}).")

        risks_and_weaknesses = []
        notice_days = int(candidate_row.get("notice_period_days", 0))
        stale_profile = float(candidate_row.get("stale_profile_risk", 0.0))
        hopping_risk = float(candidate_row.get("job_hopping_risk", 0.0))
        
        if notice_days >= 90:
            risks_and_weaknesses.append(f"Long availability timeline ({notice_days}-day notice period restriction).")
        elif notice_days >= 60:
            risks_and_weaknesses.append(f"Standard {notice_days}-day transitional notice period constraint.")
            
        if stale_profile >= 0.65:
            risks_and_weaknesses.append("Profile latency risk: low interaction footprint on the platform in recent months.")
            
        if hopping_risk >= 0.80:
            risks_and_weaknesses.append("Job stability flags: frequency of short-term employments suggests high transition risk.")
            
        if missing_skills:
            risks_and_weaknesses.append(f"Skill gap identified: lacks verified portfolio validation in {', '.join(missing_skills[:3])}.")

        # Define threshold defaults
        t_exceptional = 0.85
        t_recommended = 0.72
        t_caution = 0.58

        try:
            import json
            from pathlib import Path
            proj_root = Path(__file__).resolve().parent.parent
            config_path = proj_root / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    use_fallback = cfg.get("semantic_matcher", {}).get("use_fallback", False)
                if use_fallback:
                    t_exceptional = 0.60
                    t_recommended = 0.50
                    t_caution = 0.40
        except Exception:
            pass

        rec = "NOT ALIGNED"
        hiring_action = "Reject profile or place in alternative pipelines."
        
        if final_score >= t_exceptional:
            rec = "EXCEPTIONAL"
            hiring_action = (
                f"Schedule immediate hiring reviews. Candidate satisfies {len(matched_skills)} core technical "
                "requirements with high platform responsiveness metrics."
            )
        elif final_score >= t_recommended:
            rec = "RECOMMENDED"
            hiring_action = (
                f"Progress to screening interviews. Solid alignment with '{current_title}' parameters. "
                "Evaluate notice timeline constraints early in discussion."
            )
        elif final_score >= t_caution:
            rec = "CONSIDER_WITH_CAUTION"
            hiring_action = f"Schedule screening call only if secondary candidates are sparse. Review skill gaps of: {', '.join(missing_skills[:2])}."

        score_breakdown = {
            "overall_confidence": final_score,
            "semantic_matching": float(candidate_row.get("composite_semantic_score", 0.0)),
            "profile_quality": float(candidate_row.get("composite_quality_score", 0.5)),
            "recruitability": float(candidate_row.get("composite_recruitability_score", 0.5)),
            "associated_risk": float(candidate_row.get("aggregate_risk_score", 0.0))
        }

        reasons_list = []
        if matched_skills:
            reasons_list.append(f"Matched {len(matched_skills)} required tech skills")
        if response_rate >= 0.60:
            reasons_list.append("high collaborator profile")
        if notice_days <= 30:
            reasons_list.append("immediate/30-day availability window")
        if stability >= 0.80:
            reasons_list.append("stable job records")
            
        reasoning_one_liner = f"{candidate_row.get('current_title', 'Specialist')} with {candidate_row.get('years_of_experience', 0.0):.1f} yrs; {'; '.join(reasons_list[:3])}."

        return {
            "candidate_id": cand_id,
            "name": name,
            "current_title": current_title,
            "rank": int(candidate_row.get("leaderboard_rank", 999)),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "risks_and_weaknesses": risks_and_weaknesses,
            "recommendation": rec,
            "hiring_action_recommendation": hiring_action,
            "one_liner_reasoning": reasoning_one_liner,
            "score_breakdown": score_breakdown
        }

    def generate_formatted_markdown_report(self, cand_report: Dict[str, Any]) -> str:
        """
        Generates a clean structure report explaining candidate evaluation. Zero emojis.
        """
        breakdown = cand_report["score_breakdown"]
        
        def get_bars(score: float) -> str:
            blocks = int(round(score * 10))
            return "=" * blocks + "." * (10 - blocks)

        md = (
            f"### Evaluation Report: **{cand_report['name']}** (Rank #{cand_report['rank']})\n"
            f"**Candidate ID**: `{cand_report['candidate_id']}` | **Current Title**: *{cand_report['current_title']}*\n\n"
            f"#### Recommendation Status: **`{cand_report['recommendation']}`**\n"
            f"> **Hiring Action**: {cand_report['hiring_action_recommendation']}\n\n"
            f"----\n"
            f"#### Multi-Dimensional Score Analysis\n"
            f"| Metric Class | score | Visualization |\n"
            f"| :--- | :---: | :---: |\n"
            f"| **Hiring Confidence Score** | `{breakdown['overall_confidence']:.4f}` | `[{get_bars(breakdown['overall_confidence'])}]` |\n"
            f"| Semantic Matching Alignments | `{breakdown['semantic_matching']:.2f}` | `[{get_bars(breakdown['semantic_matching'])}]` |\n"
            f"| Career Tenure & Profile Quality | `{breakdown['profile_quality']:.2f}` | `[{get_bars(breakdown['profile_quality'])}]` |\n"
            f"| Recruitability & Engagement | `{breakdown['recruitability']:.2f}` | `[{get_bars(breakdown['recruitability'])}]` |\n"
            f"| Associated Risk Margin (Penalty) | `{breakdown['associated_risk']:.2f}` | `[{get_bars(breakdown['associated_risk'])}]` |\n\n"
            f"----\n"
            f"#### Strengths Identified\n"
        )
        
        for str_bullet in cand_report["strengths"]:
            md += f"- [STRENGTH] {str_bullet}\n"
        if not cand_report["strengths"]:
            md += "- No significant strengths recorded.\n"
            
        md += "\n#### Risks & skill Gaps Identified\n"
        for risk_bullet in cand_report["risks_and_weaknesses"]:
            md += f"- [RISK AREA] {risk_bullet}\n"
        if not cand_report["risks_and_weaknesses"]:
            md += "- Low transitional risks detected.\n"
            
        md += (
            f"\n#### Core Technical Alignments\n"
            f"- **Matched Required Skills**: `{', '.join(cand_report['matched_skills']) if cand_report['matched_skills'] else 'None'}`\n"
            f"- **Missing Skill Gaps**: `{', '.join(cand_report['missing_skills']) if cand_report['missing_skills'] else 'None'}`\n"
        )
        return md


if __name__ == "__main__":
    from data_loader import CandidateDataLoader
    from feature_engineering import RecruiterFeatureEngine
    from matching_layer import SemanticMatchingLayer
    from hybrid_ranking_engine import HybridRankingEngine

    proj_root = Path(__file__).parent.parent
    test_schema = proj_root / "data" / "candidate_schema.json"
    test_candidates = proj_root / "data" / "sample_candidates.json"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Verifying RecruiterExplainabilityEngine integration...")

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
        
        explainer = RecruiterExplainabilityEngine(target_skills_list=required_tools)
        top_cand_row = leaderboard_df.iloc[0]
        report = explainer.generate_candidate_explanation(top_cand_row, dfs)
        markdown_view = explainer.generate_formatted_markdown_report(report)
        logger.info("Explainability verification complete.")
    except Exception as e:
        logger.error(f"Verification failure: {e}", exc_info=True)
        sys.exit(1)
