# -*- coding: utf-8 -*-
"""Collect German climate-change news articles and cluster them by topic.

Pipeline:
    1. Read the climate-change RSS feeds of several German newspapers.
    2. Keep articles from the requested time window.
    3. Crawl each article page and extract its body text. Every request is
       checked against the site's robots.txt and rate-limited.
    4. Pre-process the text: remove punctuation and digits, tokenize,
       tag parts of speech, lemmatize, drop stop words. Only the resulting
       lemmas are kept; the article text itself is not stored.
    5. Turn the texts into TF-IDF vectors and group them with
       agglomerative (hierarchical) clustering.
"""

import pickle
import string
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit

import feedparser
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from germalemma import GermaLemma
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

for _resource in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(_resource, quiet=True)

# Pre-trained German part-of-speech tagger (see scripts/train_pos_tagger.py).
TAGGER_PATH = Path(__file__).parent / "nltk_german_classifier_data.pickle"

# The crawler identifies itself openly; robots.txt rules are checked for this name.
CRAWLER_NAME = "climate-news-clustering"
HTTP_HEADERS = {"User-Agent": f"Mozilla/5.0 (compatible; {CRAWLER_NAME})"}

# Minimum pause between two requests to the same site, unless robots.txt asks for more.
DEFAULT_CRAWL_DELAY = 1.0


class ArticleCollector:

    rss_urls = {
        "taz": "https://taz.de/Schwerpunkt-Klimawandel/!t5008262;rss/",
        "Spiegel": "https://www.spiegel.de/thema/klimawandel/index.rss",
        "Klimareporter": "https://www.klimareporter.de/feed/rss/",
    }

    lemmatizer = GermaLemma()
    with open(TAGGER_PATH, "rb") as f:
        tagger = pickle.load(f)
    stop_words = set(stopwords.words("german"))

    _robots: dict[str, urllib.robotparser.RobotFileParser] = {}  # one parser per site
    _last_request: dict[str, float] = {}  # time of the last request per site

    # ------------------------------------------------------------------
    # Polite crawling: robots.txt and rate limiting
    # ------------------------------------------------------------------

    @classmethod
    def _robots_for(cls, url: str) -> urllib.robotparser.RobotFileParser:
        """Load and cache the robots.txt rules of the site that hosts ``url``."""
        parts = urlsplit(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site not in cls._robots:
            parser = urllib.robotparser.RobotFileParser()
            request = urllib.request.Request(f"{site}/robots.txt", headers=HTTP_HEADERS)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    parser.parse(response.read().decode("utf-8", "replace").splitlines())
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    parser.allow_all = True      # no robots.txt: no restrictions
                else:
                    parser.disallow_all = True   # site refuses access: do not crawl
            except urllib.error.URLError:
                parser.disallow_all = True       # rules unknown: do not crawl
            cls._robots[site] = parser
        return cls._robots[site]

    @classmethod
    def _allowed_by_robots(cls, url: str) -> bool:
        return cls._robots_for(url).can_fetch(CRAWLER_NAME, url)

    @classmethod
    def _wait_for_turn(cls, url: str) -> None:
        """Pause so that one site gets at most one request per crawl delay."""
        site = urlsplit(url).netloc
        delay = cls._robots_for(url).crawl_delay(CRAWLER_NAME) or DEFAULT_CRAWL_DELAY
        elapsed = time.monotonic() - cls._last_request.get(site, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        cls._last_request[site] = time.monotonic()

    # ------------------------------------------------------------------
    # Crawling and text extraction
    # ------------------------------------------------------------------

    @classmethod
    def _fetch_page(cls, url: str) -> BeautifulSoup:
        if not cls._allowed_by_robots(url):
            raise PermissionError("disallowed by robots.txt")
        cls._wait_for_turn(url)
        request = urllib.request.Request(url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(request, timeout=30) as page:
            return BeautifulSoup(page, "html.parser")

    @staticmethod
    def _strip_taz_spiegel(soup: BeautifulSoup) -> str:
        paragraphs = soup.find("article").find_all("p")
        return " ".join(p.get_text() for p in paragraphs)

    @staticmethod
    def _strip_klimareporter(soup: BeautifulSoup) -> str:
        parts = soup.select("div.article-intro, div.article-body p")
        if not parts:
            raise ValueError("article text not found")
        return " ".join(p.get_text(" ", strip=True) for p in parts)

    @classmethod
    def _extract_text(cls, soup: BeautifulSoup, publisher: str) -> str:
        if publisher in ("taz", "Spiegel"):
            return cls._strip_taz_spiegel(soup)
        if publisher == "Klimareporter":
            return cls._strip_klimareporter(soup)
        raise ValueError(f"Unknown publisher: {publisher}")

    # ------------------------------------------------------------------
    # Text pre-processing
    # ------------------------------------------------------------------

    @classmethod
    def _pre_process_text(cls, text: str) -> str:
        """Reduce an article to a space-separated string of lemmas."""
        text = text.translate(str.maketrans("", "", string.punctuation + "0123456789„“"))
        tokens = word_tokenize(text, language="german")
        tagged = cls.tagger.tag(tokens)  # part-of-speech tagging

        lemmas = []
        for word, pos in tagged:
            try:
                lemmas.append(cls.lemmatizer.find_lemma(word, pos))
            except ValueError:  # GermaLemma only handles nouns, verbs, adjectives, adverbs
                continue
        return " ".join(w for w in lemmas if w.lower() not in cls.stop_words)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_articles_by(cls, publisher: str, time_period=pd.Timedelta(7, unit="D")) -> pd.DataFrame:
        """Fetch, filter and pre-process recent articles from one publisher."""
        if publisher not in cls.rss_urls:
            raise ValueError(f"Unknown publisher: {publisher}")
        print(f"Fetching {publisher} ...")

        # list of articles from the RSS feed
        feed_url = cls.rss_urls[publisher]
        if not cls._allowed_by_robots(feed_url):
            print(f"  feed disallowed by robots.txt, skipping {publisher}")
            return pd.DataFrame()
        cls._wait_for_turn(feed_url)
        feed = feedparser.parse(feed_url, agent=HTTP_HEADERS["User-Agent"])
        articles = pd.DataFrame(feed.entries)
        if articles.empty:
            print(f"  no entries in {publisher} feed")
            return articles
        articles["published"] = pd.to_datetime(articles["published"], utc=True).dt.tz_localize(None)

        # keep only articles from the requested time window
        age = pd.Timestamp.now(tz="UTC").tz_localize(None) - articles["published"]
        articles = articles.loc[age <= time_period, ["title", "link", "published"]].copy()
        articles["publisher"] = publisher
        articles["link"] = articles["link"].str.split("#").str[0]  # drop tracking anchors like #ref=rss

        # crawl each article and extract the text
        rows = []
        for _, article in articles.iterrows():
            try:
                soup = cls._fetch_page(article.link)
                text = cls._extract_text(soup, publisher)
            except Exception as err:  # one broken article should not stop the run
                print(f"  skipped {article.link} ({err})")
                continue
            print(f"  {article.link}")
            # keep only the lemmas needed for clustering, not the article text
            rows.append({**article.to_dict(),
                         "text_processed": cls._pre_process_text(text)})
        return pd.DataFrame(rows)

    @classmethod
    def get_articles(cls, time_period=pd.Timedelta(7, unit="D"),
                     publisher=("Klimareporter", "taz", "Spiegel")) -> pd.DataFrame:
        """Fetch recent articles from several publishers into one table."""
        frames = [cls.get_articles_by(p, time_period) for p in publisher]
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def cluster_articles(articles: pd.DataFrame, n_cluster: int | None = None,
                         verbose: bool = False) -> pd.DataFrame:
        """Add a ``cluster`` column that groups articles about the same topic.

        If ``n_cluster`` is not given, it is set to articles / publishers,
        i.e. the number of stories we would expect if every outlet reported
        on each story once.
        """
        X = TfidfVectorizer().fit_transform(articles["text_processed"])

        if not n_cluster:
            n_cluster = max(1, len(articles) // articles["publisher"].nunique())
        if verbose:
            print(f"Info: n_cluster={n_cluster}")

        hac = AgglomerativeClustering(n_clusters=n_cluster, metric="euclidean")
        articles["cluster"] = hac.fit_predict(X.toarray())
        return articles
