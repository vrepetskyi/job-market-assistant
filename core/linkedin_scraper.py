import csv
import logging
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import EventData, Events
from linkedin_jobs_scraper.filters import (
    ExperienceLevelFilters,
    IndustryFilters,
    OnSiteOrRemoteFilters,
    RelevanceFilters,
    TimeFilters,
    TypeFilters,
)
from linkedin_jobs_scraper.query import Query, QueryFilters, QueryOptions


def scrap_query(file_name, job_titles, locations) -> None:
    logging.basicConfig(level=logging.INFO)

    # Fired once for each successfully processed job
    def on_data(data: EventData):
        print(
            "[ON_DATA]",
            data.title,
            data.company,
            data.company_link,
            data.date,
            data.link,
            len(data.description),
        )

        with open(file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    data.job_id,
                    data.title,
                    data.company,
                    data.date,
                    data.link,
                    data.description,
                    data.place,
                    data.date,
                    data.insights,
                ]
            )

    # Scrape public available jobs on Linkedin using headless browser. For each job, the following fields are extracted:
    # job_id, link, apply_link, title, company, company_link, company_img_link, place, description, description_html, date, insights.
    # data.company_link, data.insights, data.insights,

    scraper = LinkedinScraper(
        headless=True,  # Overrides headless mode only if chrome_options is None
        max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
        slow_mo=0.5,  # Slow down the scraper to avoid 'Too many requests 429' errors (in seconds)
        page_load_timeout=300,  # Page load timeout (in seconds)
    )

    # Add event listeners
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, lambda error: logging.error(error))

    queries = (
        Query(
            query=job_title,
            options=QueryOptions(
                locations=locations,
                apply_link=True,  # Try to extract apply link (easy applies are skipped). If set to True, scraping is slower because an additional page must be navigated. Default to False.
                skip_promoted_jobs=True,  # Skip promoted jobs. Default to False.
                page_offset=2,  # How many pages to skip
                limit=10,
                filters=QueryFilters(
                    # company_jobs_url='https://www.linkedin.com/jobs/search/?f_C=1441%2C17876832%2C791962%2C2374003%2C18950635%2C16140%2C10440912&geoId=92000000',  # Filter by companies.
                    relevance=RelevanceFilters.RECENT,
                    time=TimeFilters.MONTH,
                    type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                    on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE],
                    experience=[
                        ExperienceLevelFilters.INTERNSHIP,
                        ExperienceLevelFilters.ENTRY_LEVEL,
                    ],
                    industry=[IndustryFilters.IT_SERVICES],
                ),
            ),
        )
        for job_title in job_titles
    )

    scraper.run(queries)


if __name__ == "__main__":
    FILE_NAME = "../data/jobs.csv"

    JOB_TITLES = (
        "Machine Learning Engineer",
        "Data Scientist",
        "Data Engineer",
        "Data Analyst",
        "AI Software Developer",
        "AI Research Scientist" "AI Engineer",
        "ML Engineer",
        "Computer Vision enginneer",
        "MLOps Developer",
        "Data Developer" "Machine Learning",
        "Deep Learning Engineer",
        "NLP developer",
        "AI full stack developer" "Python developer",
        "Artificial Intelligence",
    )

    LOCATIONS = (
        "Poland",
        "Poznan",
        "Europe",
        "Warsaw",
        "Wroclaw",
        "Bydgoszcz",
        "Prague",
        "Berlin",
        "Dublin",
        "Gdansk",
        "Szczecin",
        "Katowice",
        "Lodz",
    )

    scrap_query(FILE_NAME, JOB_TITLES, LOCATIONS)


def scrap_urls(urls: Iterable[str]) -> pd.DataFrame:
    entries = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract job title, company name, job description, and location
        tags = (
            soup.find("h1", {"class": "topcard__title"}),
            soup.find("a", {"class": "topcard__org-name-link"}),
            soup.find("div", {"class": "description__text"}),
            soup.find("span", {"class": "topcard__flavor topcard__flavor--bullet"}),
        )

        # Clear from extra spaces and new lines
        values = (tag.text.strip() if tag else "" for tag in tags)

        entries.append(values)

    return pd.DataFrame(
        entries, columns=("job_title", "company", "job_desc", "location")
    )
