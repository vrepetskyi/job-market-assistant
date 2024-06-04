import pandas as pd
from IPython.display import Markdown, display

from .models import Model
from .retriever import get_retriever, nodes_to_postings


class JobMarketAssistant:
    """
    A tool to help you find your niche in the job market.
    Generate cover letters for interesting job posting.
    Identify skill gaps based on the available offers, your CV, and the desired job title.
    """

    def __init__(
        self,
        model: Model,
        cover_letter_prompt: str,
        job_title_prompt: str,
        key_points_prompt: str,
        missing_skills_prompt: str,
    ) -> None:
        "Initialize the tool with a LLM model of choice and a set of respective prompts."

        self.model = model

        self.cover_letter_prompt = cover_letter_prompt
        self.job_title_prompt = job_title_prompt
        self.key_points_prompt = key_points_prompt
        self.missing_skills_prompt = missing_skills_prompt

    def get_cover_letter(self, job_posting: pd.Series, cv: str) -> str:
        "Generate a cover letter for a selected job posting based on a CV."

        job_title, company, job_desc, location = job_posting

        prompt = self.cover_letter_prompt.format(
            job_title=job_title,
            company=company,
            job_desc=job_desc,
            location=location,
            cv=cv,
        )

        cover_letter = self.model.complete(prompt)

        display(
            Markdown(f"### Cover letter for {job_title} at {company}:\n" + cover_letter)
        )

        return cover_letter

    def get_missing_skills(
        self,
        job_postings: pd.DataFrame,
        cv: str,
        job_title: str = None,
    ) -> str:
        """
        Get top 10 job postings matching the job title and a list of skills required for these jobs that are missing in the CV.
        If the desired job title is not provided it will be generated automatically from the CV.
        """

        final_job_title = job_title
        if not job_title:
            final_job_title = self.model.complete(self.job_title_prompt.format(cv=cv))

        retriever = get_retriever(job_postings)
        nodes = retriever.retrieve(final_job_title)
        top_10 = nodes_to_postings(nodes)

        display(
            Markdown(
                f'### Postings best matching job title {final_job_title}{"" if job_title else " (derived from the CV)"}'
            )
        )
        display(top_10)

        # Prompt the model to extract the keypoints from a single retrived job posting
        key_points = "\n\n".join(
            self.model.complete(
                self.key_points_prompt.format(
                    job_title=job_title,
                    company=company,
                    job_desc=job_desc,
                    location=location,
                )
            )
            for _, job_title, company, job_desc, location, _ in top_10.itertuples()
        )

        # Compose the main prompt based on the extracted key_points and cv
        missing_skills = self.model.complete(
            self.missing_skills_prompt.format(key_points=key_points, cv=cv)
        )

        display(Markdown("### Missing skills:\n" + missing_skills))

        return missing_skills
