import random
from types import SimpleNamespace

from conftest import load_module

scraper = load_module("scraping/src/main.py", "scraping_main")
RandomList = load_module("scraping/src/random_list.py", "scraping_random_list").RandomList

GOOD = "This paragraph is long enough to be kept because it is clearly prose. " * 3


def test_get_paragraphs_filters_and_stops_at_end_section():
    text = "\n".join(
        [
            GOOD,
            "too short.",
            "lowercase start " + GOOD,
            GOOD + " doi:10.1000/xyz.",
            "A bullet-like line without a full stop " + "x" * 120,
            "References",
            GOOD,
        ]
    )
    page = SimpleNamespace(text=text)
    assert scraper.get_paragraphs(page) == [GOOD.strip()]


def test_random_list_pops_every_item_once():
    random.seed(0)
    items = RandomList(range(20))
    popped = []
    while items:
        popped.append(items.pop())
    assert sorted(popped) == list(range(20))
