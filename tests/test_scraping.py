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


def test_scraper_timeout_handler_matches_the_pinned_wikipedia_api():
    # Regression: requirements allowed wikipedia-api >= 0.9, which dropped requests,
    # so `from requests.exceptions import Timeout` failed and timeouts were never caught.
    from importlib.metadata import requires, version

    import requests

    assert scraper.Timeout is requests.exceptions.Timeout
    major, minor = (int(x) for x in version("wikipedia-api").split(".")[:2])
    assert (major, minor) < (0, 9)
    assert any(r.startswith("requests") for r in requires("wikipedia-api") or [])
