"""Collect last week's climate news from German newspapers and group them by topic.

Usage:
    python cluster_news.py

Writes result.csv (cluster, publisher, title, link) and prints each cluster.
"""

import pandas as pd

from article_collector.article_collector import ArticleCollector

articles = ArticleCollector.get_articles(
    time_period=pd.Timedelta(7, unit="D"),
    publisher=["taz", "Spiegel", "Klimareporter"],
)
articles = ArticleCollector.cluster_articles(articles, verbose=True)

result = articles[["cluster", "publisher", "title", "link"]].sort_values("cluster")
result.to_csv("result.csv", index=False, encoding="utf-8")

for cluster, group in result.groupby("cluster"):
    print(f"\n--- Cluster {cluster} ---")
    for _, row in group.iterrows():
        print(f"[{row.publisher}] {row.title}")
        print(f"    {row.link}")
