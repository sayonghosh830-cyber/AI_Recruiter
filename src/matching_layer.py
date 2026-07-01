import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("MatchingLayer")

try:
    import torch
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info("Sentence transformers library not loaded. Falling back to TF-IDF.")


class SkillExtractor:
    """
    Scans candidate profile summary, headline, and career history descriptions
    to extract technical skills based on a configurable skill dictionary.
    """
    def __init__(self, skill_dictionary: List[str]):
        self.skill_dictionary = skill_dictionary
        self.skill_map = {}
        for s in skill_dictionary:
            name_strip = s.strip()
            if name_strip:
                self.skill_map[name_strip.lower()] = name_strip

    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extracts skills from text based on self.skill_dictionary.
        Uses exact substring or word-boundaries to avoid false positives.
        """
        if not isinstance(text, str) or not text:
            return []
        
        text_lower = text.lower()
        extracted = []
        
        for k_lower, original_name in self.skill_map.items():
            if re.search(r'[^a-zA-Z0-9]', k_lower):
                # If there are special characters (e.g. C++, .NET, C#), do substring matching
                if k_lower in text_lower:
                    extracted.append(original_name)
            else:
                # Word boundary matching for clean alphanumeric matches
                pattern = rf"\b{re.escape(k_lower)}\b"
                if re.search(pattern, text_lower):
                    extracted.append(original_name)
                    
        return list(set(extracted))

    def enrich_skills_dataframe(self, df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Enriches and updates the relational skills table by scanning profiles (summary/headline)
        and career histories (description/title) and appending extracted candidate skills.
        """
        profiles_df = df_dict["profiles"]
        careers_df = df_dict["careers"]
        skills_df = df_dict["skills"].copy()

        # Existing candidate-skill sets
        existing_pairs = set()
        if not skills_df.empty:
            for _, row in skills_df.iterrows():
                c_id = row["candidate_id"]
                s_name = str(row["skill_name"]).strip().lower()
                existing_pairs.add((c_id, s_name))

        extracted_records = []

        # 1. Scan Profile headlines and summaries
        for _, row in profiles_df.iterrows():
            cand_id = row["candidate_id"]
            headline = str(row.get("headline", ""))
            summary = str(row.get("summary", ""))
            combined = f"{headline} {summary}"
            
            ext_skills = self.extract_skills_from_text(combined)
            for skill in ext_skills:
                skill_lower = skill.lower()
                pair = (cand_id, skill_lower)
                if pair not in existing_pairs:
                    existing_pairs.add(pair)
                    extracted_records.append({
                        "candidate_id": cand_id,
                        "skill_name": skill,
                        "proficiency": "extracted",
                        "endorsements_count": 0,
                        "usage_months": None
                    })

        # 2. Scan Career History titles and descriptions
        for _, row in careers_df.iterrows():
            cand_id = row["candidate_id"]
            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            combined = f"{title} {desc}"
            duration = row.get("duration_months", 0)
            duration_val = int(duration) if pd.notna(duration) else None

            ext_skills = self.extract_skills_from_text(combined)
            for skill in ext_skills:
                skill_lower = skill.lower()
                pair = (cand_id, skill_lower)
                if pair not in existing_pairs:
                    existing_pairs.add(pair)
                    extracted_records.append({
                        "candidate_id": cand_id,
                        "skill_name": skill,
                        "proficiency": "extracted_career",
                        "endorsements_count": 0,
                        "usage_months": duration_val
                    })

        if extracted_records:
            new_skills_df = pd.DataFrame(extracted_records)
            skills_df = pd.concat([skills_df, new_skills_df], ignore_index=True)
            logger.info(f"Enriched candidate skills: added {len(extracted_records)} newly extracted skills from resume text scanning.")
        else:
            logger.info("No additional skills extracted from profile or career text scans.")

        return skills_df



class CandidateDocumentSynthesizer:
    """
    Synthesizes complete textual views (synthetic documents) of resumes or candidates,
    amalgamating demographics, historical roles, skills lists, and academics.
    """

    @staticmethod
    def synthesize_candidate_doc(candidate_row: pd.Series, df_dict: Dict[str, pd.DataFrame]) -> str:
        """
        Groups nested records together into a clean, cohesive summary passage.
        """
        cand_id = candidate_row["candidate_id"]
        
        headline = str(candidate_row.get("headline", "")).strip()
        summary = str(candidate_row.get("summary", "")).strip()
        current_title = str(candidate_row.get("current_title", "")).strip()
        current_company = str(candidate_row.get("current_company", "")).strip()
        current_industry = str(candidate_row.get("current_industry", "")).strip()
        years_exp = f"{candidate_row.get('years_of_experience', 0.0):.1f} years of experience"

        skills_df = df_dict["skills"]
        cand_skills = skills_df[skills_df["candidate_id"] == cand_id]
        skills_token_list = []
        for _, skill_row in cand_skills.iterrows():
            skills_token_list.append(f"{skill_row['skill_name']} ({skill_row['proficiency'] or 'intermediate'})")
        skills_string = ", ".join(skills_token_list) if skills_token_list else "No listed skills"

        careers_df = df_dict["careers"]
        cand_careers = careers_df[careers_df["candidate_id"] == cand_id]
        career_list = []
        for _, job in cand_careers.iterrows():
            title = str(job.get("title", ""))
            company = str(job.get("company", ""))
            desc = str(job.get("description", "")).strip()
            duration = f"{job.get('duration_months', 0)} months"
            career_list.append(f"Role: {title} at {company} (Duration: {duration}). Responsibilities: {desc}")
        career_history_string = " | ".join(career_list) if career_list else "No listed work milestones"

        education_df = df_dict["education"]
        cand_edu = education_df[education_df["candidate_id"] == cand_id]
        edu_list = []
        for _, edu in cand_edu.iterrows():
            edu_list.append(f"{edu.get('degree', 'Degree')} in {edu.get('field_of_study', 'Field')} from {edu.get('institution', 'University')}")
        education_string = ", ".join(edu_list) if edu_list else "No listed degrees"

        doc = (
            f"Candidate ID: {cand_id}\n"
            f"Current Position: {current_title} at {current_company} in {current_industry} industry. Total seniority: {years_exp}.\n"
            f"Headline Summary: {headline}. Profile overview: {summary}\n"
            f"Technical Skills and Capabilities: {skills_string}\n"
            f"Professional Job Milestones and Career History: {career_history_string}\n"
            f"Scholastic Background and Credentials: {education_string}"
        )
        doc = re.sub(r"\s+", " ", doc)
        return doc


class SemanticMatchingLayer:
    """
    Evaluates applicant suitability score parameters against job descriptions using
    semantic embedding searches and hard-skill overlap evaluations.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_fallback: bool = False):
        self.model_name = model_name
        
        # Determine fallback state and exact reason
        fallback_reason = None
        if use_fallback:
            fallback_reason = "Explicitly configured in config.json (use_fallback=true)"
        elif not SENTENCE_TRANSFORMERS_AVAILABLE:
            fallback_reason = "sentence-transformers or torch library is not available in the current Python environment"
            
        self.use_fallback = use_fallback or (not SENTENCE_TRANSFORMERS_AVAILABLE)
        self.model: Optional[Any] = None
        self.fallback_vectorizer: Optional[TfidfVectorizer] = None
        
        if not self.use_fallback:
            try:
                logger.info(f"Attempting to load sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("Pipeline is running in: SENTENCE_TRANSFORMER mode")
                logger.info(f"Loaded sentence transformer: {model_name}")
            except Exception as e:
                fallback_reason = f"SentenceTransformer model failed to load or download: {e}"
                logger.warning(f"Error loading model {model_name}: {e}. Initializing TF-IDF fallback.")
                self.use_fallback = True

        if self.use_fallback:
            logger.info("Pipeline is running in: TF_IDF_FALLBACK mode")
            if fallback_reason:
                logger.info(f"Reason for fallback mode activation: {fallback_reason}")
            self.fallback_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
                token_pattern=r"(?u)\b\w\w+\b"
            )

    def generate_candidate_documents(self, df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Synthesizes composite representation passages for profiles.
        """
        profiles_df = df_dict["profiles"].copy()
        profiles_df["synthetic_document"] = profiles_df.apply(
            lambda row: CandidateDocumentSynthesizer.synthesize_candidate_doc(row, df_dict),
            axis=1
        )
        return profiles_df

    def compute_matching_scores(
        self, 
        job_description: str, 
        target_title: str,
        target_skills: List[str],
        candidates_enriched_df: pd.DataFrame,
        df_dict: Dict[str, pd.DataFrame],
        target_job: Optional[str] = None,
        required_toolkit: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Runs multidimensional target matching against specified role requirements.
        Dimensions:
            1. semantic_similarity_score: Vector cosine match of the aggregated synthetic text.
            2. title_match_score: Direct title keyword proximity alignments.
            3. skill_overlap_score: Intersection-over-required skills overlap ratio.
        """
        if target_job is not None:
            target_title = target_job
        if required_toolkit is not None:
            target_skills = required_toolkit

        results_df = candidates_enriched_df.copy()
        candidate_docs = results_df["synthetic_document"].tolist()

        if not candidate_docs:
            logger.warning("No candidate profiles provided for matching.")
            results_df["semantic_similarity_score"] = 0.0
            results_df["title_match_score"] = 0.0
            results_df["skill_overlap_score"] = 0.0
            results_df["composite_semantic_score"] = 0.0
            return results_df

        if self.use_fallback:
            all_texts = [job_description] + candidate_docs
            tfidf_matrices = self.fallback_vectorizer.fit_transform(all_texts)
            job_vector = tfidf_matrices[0]
            candidate_vectors = tfidf_matrices[1:]
            
            global_similarities = cosine_similarity(candidate_vectors, job_vector).flatten()
        else:
            job_embedding = self.model.encode(job_description, convert_to_tensor=True)
            candidate_embeddings = self.model.encode(candidate_docs, convert_to_tensor=True, show_progress_bar=False)
            
            if hasattr(job_embedding, "cpu"):
                cos_sim_tensor = torch.nn.functional.cosine_similarity(candidate_embeddings, job_embedding.unsqueeze(0), dim=1)
                global_similarities = cos_sim_tensor.cpu().numpy()
            else:
                global_similarities = cosine_similarity(
                    candidate_embeddings.reshape(len(candidate_docs), -1),
                    job_embedding.reshape(1, -1)
                ).flatten()

        results_df["semantic_similarity_score"] = global_similarities.astype(float)

        title_matches = []
        for idx, row in results_df.iterrows():
            current_t = str(row.get("current_title", "")).strip().lower()
            headline_t = str(row.get("headline", "")).strip().lower()
            target_t = target_title.strip().lower()
            
            if target_t in current_t or current_t in target_t:
                match_score = 1.0
            elif any(word in current_t for word in target_t.split() if len(word) > 3):
                match_score = 0.7
            elif any(word in headline_t for word in target_t.split() if len(word) > 3):
                match_score = 0.5
            else:
                match_score = 0.1
            title_matches.append(match_score)
            
        results_df["title_match_score"] = title_matches

        target_skills_norm = [s.strip().lower() for s in target_skills if s.strip()]
        skill_overlaps = []
        skills_df = df_dict["skills"]
        
        for cand_id in results_df["candidate_id"]:
            cand_skills_rows = skills_df[skills_df["candidate_id"] == cand_id]
            cand_skills_items = [str(s).strip().lower() for s in cand_skills_rows["skill_name"].tolist() if pd.notna(s)]
            
            if not target_skills_norm or len(target_skills_norm) == 0:
                overlap_pct = 0.0
            else:
                matched_skills = set(target_skills_norm).intersection(set(cand_skills_items))
                overlap_pct = len(matched_skills) / len(target_skills_norm)
                
            skill_overlaps.append(overlap_pct)
            
        results_df["skill_overlap_score"] = skill_overlaps

        results_df["composite_semantic_score"] = (
            (results_df["semantic_similarity_score"] * 0.30) +
            (results_df["title_match_score"] * 0.20) +
            (results_df["skill_overlap_score"] * 0.50)
        )

        return results_df

    def compute_skill_gaps(
        self, 
        target_skills: List[str], 
        cand_id: str, 
        df_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[str]]:
        """
        Parses missing and matching skills.
        """
        target_skills_norm = [s.strip().lower() for s in target_skills if s.strip()]
        if not target_skills_norm:
            return {"matched": [], "missing": []}

        skills_df = df_dict["skills"]
        cand_skills_rows = skills_df[skills_df["candidate_id"] == cand_id]
        cand_skills_items = [str(s).strip().lower() for s in cand_skills_rows["skill_name"].tolist() if pd.notna(s)]
        cand_skills_names_dict = {str(s).lower(): str(s) for s in cand_skills_rows["skill_name"].tolist() if pd.notna(s)}

        matched = []
        missing = []
        
        for raw_req in target_skills:
            req_norm = raw_req.strip().lower()
            found = False
            for cand_skill_norm in cand_skills_items:
                if req_norm == cand_skill_norm or req_norm in cand_skill_norm or cand_skill_norm in req_norm:
                    original_name = cand_skills_names_dict.get(cand_skill_norm, raw_req)
                    matched.append(original_name)
                    found = True
                    break
            if not found:
                missing.append(raw_req)

        return {
            "matched": list(set(matched)),
            "missing": list(set(missing))
        }


if __name__ == "__main__":
    from data_loader import CandidateDataLoader
    from feature_engineering import RecruiterFeatureEngine
    
    proj_root = Path(__file__).parent.parent
    test_schema = proj_root / "data" / "candidate_schema.json"
    test_candidates = proj_root / "data" / "sample_candidates.json"
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Verifying SemanticMatchingLayer execution...")
    
    try:
        loader = CandidateDataLoader(schema_path=test_schema)
        raw_list = loader.load_candidates_raw(test_candidates)
        dfs = loader.parse_to_relational_dataframes(raw_list)
        
        engine = RecruiterFeatureEngine()
        enriched_profiles = engine.engineer_candidate_features(dfs)
        
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
        logger.info("Matching layer verification complete")
    except Exception as e:
        logger.error(f"Matching layer verification failed: {e}", exc_info=True)
        sys.exit(1)
