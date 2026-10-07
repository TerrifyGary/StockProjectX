from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient


def connect_collection(uri: str, database_name: str) -> tuple[MongoClient, Any]:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    collection = client[database_name]["articles"]
    collection.create_index([("article_key", ASCENDING)], unique=True)
    collection.create_index([("published_at", DESCENDING)])
    collection.create_index(
        [("markets", ASCENDING), ("published_at", DESCENDING)]
    )
    collection.create_index(
        [("entities.id", ASCENDING), ("published_at", DESCENDING)]
    )
    collection.create_index(
        [("language", ASCENDING), ("published_at", DESCENDING)]
    )
    client[database_name]["brief_history"].create_index(
        [("report_date", ASCENDING)], unique=True
    )
    return client, collection
