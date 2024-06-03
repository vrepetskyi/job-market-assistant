from typing import Iterable

import pandas as pd
from IPython.display import Markdown, display

from .helpers import get_retriever, nodes_to_postings
from .linkedin_scraping import scrap_urls
from .models import Model


class JobMarketAssistant:
    def __init__(
        self,
        model: Model,
        cover_letter_prompt: str,
        dream_job_prompt: str,
        key_points_prompt: str,
        missing_skills_prompt: str,
        postprocess_prompt: str = None,
    ) -> None:
        self.model = model
        self.cover_letter_prompt = cover_letter_prompt
        self.dream_job_prompt = dream_job_prompt
        self.key_points_prompt = key_points_prompt
        self.missing_skills_prompt = missing_skills_prompt
        self.postprocess_prompt = postprocess_prompt

    def scrap_job_postings_from_linkedin(urls: Iterable[str]) -> pd.DataFrame:
        return scrap_urls(urls)

    def get_cover_letter(self, job_posting: pd.Series, cv: str) -> str:
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
        top_10: pd.DataFrame,
        cv: str,
        dream_job: str = None,
    ) -> str:
        final_dream_job = dream_job
        if not dream_job:
            final_dream_job = self.model.complete(self.dream_job_prompt.format(cv=cv))

        retriever = get_retriever(top_10)
        nodes = retriever.retrieve(final_dream_job)
        top_10 = nodes_to_postings(nodes)

        display(
            Markdown(
                f'### Postings best matching job title {final_dream_job}{"" if dream_job else ", which was derived from the CV"}'
            )
        )
        display(top_10)

        key_points = "\n\n".join(
            self.model.complete(
                self.key_points_prompt.format(
                    index=index,
                    job_title=job_title,
                    company=company,
                    job_desc=job_desc,
                    location=location,
                    score=score,
                )
            )
            for index, job_title, company, job_desc, location, score in top_10.itertuples()
        )

        missing_skills = self.model.complete(
            self.missing_skills_prompt.format(cv=cv, key_points=key_points)
        )

        if self.postprocess_prompt:
            missing_skills = self.model.complete(
                self.postprocess_prompt.format(missing_skills=missing_skills)
            )

        display(Markdown("### Missing skills:\n" + missing_skills))

        return missing_skills
