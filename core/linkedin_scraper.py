from typing import Generator, Iterable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import EventData, Events
from linkedin_jobs_scraper.query import Query


def scrap_queries(queries: List[Query]) -> pd.DataFrame:
    "Scrape publicly available LinkedIn job postings matching the queries."

    # Use a headless browser
    scraper = LinkedinScraper(
        # Limit the number of threads to 1
        max_workers=1,
        # Increase the maximal page load waiting time to 300 seconds
        page_load_timeout=300,
        # Increase the delay between request to 3 seconds,
        slow_mo=3,
    )

    entries: List[Tuple[str, str, str, str]] = []

    # Store each successfully processed posting in a list
    def on_data(data: EventData):
        print(f"{len(entries) + 1}: {data.title} @ {data.company}")
        entries.append(
            (
                data.title,
                data.company,
                data.description,
                data.place,
            )
        )

    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, lambda error: print(error))
    scraper.run(queries)

    job_postings = pd.DataFrame(
        entries, columns=("job_title", "company", "job_desc", "location")
    )

    return job_postings


def scrap_urls(urls: Iterable[str]) -> pd.DataFrame:
    "Scrape publicly available LinkedIn job postings by their URLs."

    entries: List[Generator[str, None, None]] = []

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
