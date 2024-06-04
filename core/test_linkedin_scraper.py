import unittest

import linkedin_scraper
import pandas as pd
from linkedin_jobs_scraper.query import Query, QueryOptions


class TestScraping(unittest.TestCase):
    def test_scrap_url(self):
        job_postings = linkedin_scraper.scrap_urls(
            (
                "https://www.linkedin.com/jobs/view/3937215366/?alternateChannel=search&refId=pyO8PX6NFy5hyGNWQ5uRxQ%3D%3D&trackingId=aNQSGBRWWobujlSYEpbvaQ%3D%3D",
                "https://www.linkedin.com/jobs/view/3931218091/?alternateChannel=search&refId=%2BwGMIZDUEBj8w%2BcBBmKX0g%3D%3D&trackingId=MKGRxpSmYvT7yYfc0vnezg%3D%3D",
                "https://www.linkedin.com/jobs/view/3939354205/?alternateChannel=search&refId=jzm6ENFRWaolwGXJgicMTw%3D%3D&trackingId=Ryft9wMmyemu5LZP3YMVkQ%3D%3D&trk=d_flagship3_search_srp_jobs",
            )
        )

        self.assertTrue(type(job_postings), pd.DataFrame)
        self.assertEqual(len(job_postings), 3)
        self.assertTrue(
            all(
                job_postings.columns == ["job_title", "company", "job_desc", "location"]
            )
        )

    def test_scrap_query(self):
        job_postings = linkedin_scraper.scrap_queries(
            [Query("ML Engineer", options=QueryOptions(3))]
        )

        self.assertTrue(type(job_postings), pd.DataFrame)
        self.assertEqual(len(job_postings), 3)
        self.assertTrue(
            all(
                job_postings.columns == ["job_title", "company", "job_desc", "location"]
            )
        )


if __name__ == "__main__":
    unittest.main()
