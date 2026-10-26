# Climate News Topic Clustering

**Which climate stories are German newspapers covering this week, and which articles are about the same story?**

This script collects last week's climate-change articles from German online
newspapers and groups them by topic. It downloads each article, processes the
German text with NLP methods and clusters the articles. When different outlets
cover the same event, such as a new UN report or a coral reef study, their
articles end up in the same cluster.

**At a glance**

- End-to-end pipeline: from RSS feeds to topic clusters, with no manual steps
- Responsible web crawling: robots.txt is checked before every request,
  requests are rate-limited, and the crawler identifies itself openly
- Publisher-specific text extraction
- No article text is stored: each text is reduced to lemmas in memory and
  then discarded; the output contains only titles and links
- German-language NLP: tokenization, part-of-speech tagging with a tagger
  trained on the TIGER corpus, lemmatization, stop-word removal
- Unsupervised machine learning: TF-IDF vectors and hierarchical clustering
- **Stack:** Python, pandas, scikit-learn, NLTK, GermaLemma, BeautifulSoup, feedparser

## Example output

A run from October 2026 (excerpt). Each block is one detected story. The
articles in it come from different newspapers but cover the same event:

```
--- Cluster 3 ---
[taz] Wetterphänomen El Niño: El Niño lässt Argentiniens Flüsse über die Ufer treten
    http://www.taz.de/Wetterphaenomen-El-Nino/!6223706/
[Spiegel] Uno: El Niño könnte stärkster seit 40 Jahren sein – oder schlimmer
    https://www.spiegel.de/wissenschaft/uno-el-nino-koennte-staerkster-seit-40-jahren-sein-oder-schlimmer-a-ac7a789b-bc22-41ab-b87a-68cd4e416f45

--- Cluster 5 ---
[taz] Naturschutzkonferenz in Jerewan: Den Korallen und ihren Verbündeten wird es zu warm
    http://www.taz.de/Naturschutzkonferenz-in-Jerewan/!6223688/
[Spiegel] Korallenriffe: Fast alle sind laut neuer Untersuchung bedroht
    https://www.spiegel.de/wissenschaft/korallenriffe-fast-alle-sind-laut-neuer-untersuchung-bedroht-a-60ae7e2c-441f-464b-b0ab-84f6148b5b62
[taz] Globaler Korallenbericht: Drohendes Ende eines ökologischen Dienstleisters
    http://www.taz.de/Fuer-Menschen-und-Tiere-lebenswichtige-Korallenriffe-werden-absterben---die-Gesellschaft-interessiert-es-nicht/!6218350/

--- Cluster 6 ---
[taz] UN-Bericht zur Klimakrise: Menschheit „gefährlich weit“ von 1,5 Grad entfernt
    http://www.taz.de/UN-Bericht-Menschheit-gefaehrlich-weit-von-15-Grad-entfernt/!6223118/
[Spiegel] Klimakrise: Erde steuert tatsächlich auf Erwärmung um 2,8 Grad zu – Uno sieht USA als Treiber
    https://www.spiegel.de/wissenschaft/klimakrise-erde-steuert-tatsaechlich-auf-erwaermung-um-2-8-grad-zu-uno-sieht-usa-als-treiber-a-19db1fbf-f48c-407a-950b-543ec47b182c
[taz] UN-Bericht zur Klimakrise: 180-Grad-Wende angestrebt
    http://www.taz.de/UN-Bericht-zur-Klimakrise/!6223341/
[Klimareporter] Auf dem Weg zu 2,8 Grad Erwärmung
    https://klimareporter.de/international/auf-dem-weg-zu-2-8-grad-erwaermung
```

Sources: [taz](https://taz.de), [Spiegel](https://www.spiegel.de) and
[Klimareporter°](https://www.klimareporter.de).

## How it works

```
RSS feeds ─► Filter by date ─► Crawl article ──► Extract text ─► NLP pre-processing ─► TF-IDF ─► Clustering ─► result.csv
```

| Step | What happens | Tools |
| --- | --- | --- |
| Collect | Reads the climate-change RSS feed of each newspaper and keeps articles from the last 7 days | feedparser, pandas |
| Crawl | Checks the site's robots.txt, waits at least one second between requests to the same site, visits each article page and extracts the body text with a parser for each site | urllib, robotparser, BeautifulSoup |
| Pre-process | Removes punctuation and digits, tokenizes, tags parts of speech, reduces words to their base form (*Wälder* → *Wald*), removes stop words. Only these lemmas are kept; the article text is discarded | NLTK, GermaLemma, custom POS tagger |
| Vectorize | Represents each article by its TF-IDF word weights | scikit-learn |
| Cluster | Groups similar articles with agglomerative (hierarchical) clustering | scikit-learn |

**Why part-of-speech tagging?** German has rich inflection. The lemmatizer
needs to know whether a word is a noun, verb or adjective to find its base
form. Without it, *Klimaziel* and *Klimazielen* would count as two different
words. The tagger was trained on the TIGER corpus, a large annotated corpus of
German newspaper text ([`scripts/train_pos_tagger.py`](scripts/train_pos_tagger.py)).

**How many clusters?** Unless a number is given, the script uses
*articles / newspapers*. This is the number of stories you would get if every
outlet covered each story once. It is a simple heuristic that works well enough
for a weekly news cycle.

## Repository layout

| Path | Content |
| --- | --- |
| [`cluster_news.py`](cluster_news.py) | Entry point: collects, clusters and writes `result.csv` |
| [`article_collector/`](article_collector/) | The pipeline (`ArticleCollector` class) and the trained POS tagger |
| [`ClassifierBasedGermanTagger/`](ClassifierBasedGermanTagger/) | Third-party tagger class by Philipp Nolte (Apache 2.0), needed to load the trained model. Only change: regex strings made raw strings for current Python |
| [`scripts/train_pos_tagger.py`](scripts/train_pos_tagger.py) | Retrains the POS tagger from the TIGER corpus (corpus not included, its license does not allow redistribution) |

## Running it

```bash
pip install -r requirements.txt
python cluster_news.py
```

The script prints the clusters and writes them to `result.csv`. A run takes about a
minute, mostly because the crawler pauses between requests.

## Legal and ethical crawling

This is a research and demonstration tool for non-commercial use. It is
designed to put as little load as possible on the publishers and to keep
none of their content:

- **robots.txt is respected.** Before each request the crawler checks the
  site's robots.txt for its own name (`climate-news-clustering`) and skips
  any page it is not allowed to visit. If robots.txt cannot be read, the site
  is not crawled.
- **Low load.** At most one request per second per site, or slower if
  robots.txt sets a crawl delay. Only articles from the last seven days are
  visited.
- **Open identification.** The crawler sends its own user-agent name and does
  not pretend to be a browser.
- **No copies of articles.** Article texts exist only in memory, are reduced
  to word lists for clustering and then discarded. The output contains only
  titles and links, which the publishers themselves distribute in their RSS
  feeds.
- **Publishers who refuse are excluded.** Zeit Online blocks automated access
  to its articles and is therefore not included.

Some publishers, including taz and Spiegel, state in their robots.txt that
they reserve the right to text and data mining (§ 44b UrhG) and do not permit
automated access without their consent. German copyright law allows text and
data mining for non-commercial scientific research (§ 60d UrhG), which is the
context this tool was built for. Anyone who runs it for other purposes should
check each publisher's terms and ask for permission where needed.


## Use of AI tools

Parts of this repository were cleaned up and documented with the help of the
AI coding assistant Claude Code. The original pipeline, the methods and the
design are my own.
