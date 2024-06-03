import csv
import os
import unittest

from linkedin_scraping import scrap_query

FILE_NAME = "jobs_test.csv"
COL_NUMBER = 9


class TestScraping(unittest.TestCase):
    def setUp(self):
        scrap_query(FILE_NAME, True)

    def tearDown(self):
        os.remove(FILE_NAME)

    def test_file_creation(self):
        self.assertTrue(os.path.exists(FILE_NAME), "File was not created.")

    def test_file_contents(self):
        with open(FILE_NAME, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.assertEqual(
                    len(row), COL_NUMBER, "Invalid number of columns in the file."
                )


if __name__ == "__main__":
    unittest.main()
