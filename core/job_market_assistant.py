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
        job_title_prompt: str,
        key_points_prompt: str,
        missing_skills_prompt: str,
    ) -> None:
        self.model = model
        self.cover_letter_prompt = cover_letter_prompt
        self.job_title_prompt = job_title_prompt
        self.key_points_prompt = key_points_prompt
        self.missing_skills_prompt = missing_skills_prompt

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
        job_postings: pd.DataFrame,
        cv: str,
        job_title: str = None,
    ) -> str:
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

        missing_skills = self.model.complete(
            self.missing_skills_prompt.format(key_points=key_points, cv=cv)
        )

        display(Markdown("### Missing skills:\n" + missing_skills))

        return missing_skills
