from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stock_news.collector import build_gdelt_feeds, collect_feed
from stock_news.config import Settings, load_companies, load_feeds, load_topics, load_yaml
from stock_news.daily_briefs import fetch_price_snapshots, make_daily_brief
from stock_news.forecast_tracking import pending_price_start_dates
from stock_news.storage import connect_collection
from stock_news.summarizer import LocalMTS5Summarizer
from stock_news.translation import LocalMarianTranslator, backfill_english_translations
from stock_news.news_window import latest_completed_window
from stock_news.mediacloud_source import build_mediacloud_feeds, fetch_mediacloud_entries


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Collect, summarize, and store multilingual stock-related news."
    )
    parser.add_argument("--feeds", type=Path, default=Path("config/sources.yaml"))
    parser.add_argument("--watchlist", type=Path, default=Path("config/watchlist.yaml"))
    parser.add_argument("--topics", type=Path, default=Path("config/topics.yaml"))
    parser.add_argument(
        "--backfill-english",
        action="store_true",
        help="Translate missing English title and summary fields in existing MongoDB records.",
    )
    parser.add_argument("--window-start", help="ISO timestamp for the inclusive news window start.")
    parser.add_argument("--window-end", help="ISO timestamp for the exclusive news window end.")
    args = parser.parse_args()

    source_config = load_yaml(args.feeds)
    companies = load_companies(args.watchlist)
    topics = load_topics(args.topics)
    feeds = load_feeds(args.feeds)
    feeds.extend(build_gdelt_feeds(source_config.get("gdelt", {}), companies, topics))
    feeds.extend(
        build_mediacloud_feeds(
            source_config.get("mediacloud", {}), companies, topics
        )
    )
    if not feeds:
        parser.error(
            f"No news sources are enabled in {args.feeds}. Enable GDELT or Media Cloud, "
            "or add RSS/Atom feeds."
        )

    settings = Settings.from_environment()
    if bool(args.window_start) != bool(args.window_end):
        parser.error("--window-start and --window-end must be supplied together")
    if args.window_start:
        window_start = datetime.fromisoformat(args.window_start.replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(args.window_end.replace("Z", "+00:00"))
        if window_start.tzinfo is None or window_end.tzinfo is None:
            parser.error("Window timestamps must include a timezone")
        window_start = window_start.astimezone(timezone.utc)
        window_end = window_end.astimezone(timezone.utc)
    else:
        window_start, window_end = latest_completed_window(
            settings.window_timezone, settings.window_start_hour,
            window_hours=settings.window_hours,
        )
    client, collection = connect_collection(settings.mongo_uri, settings.mongo_database)
    try:
        collection.database.client.admin.command("ping")
        if args.backfill_english:
            translator = LocalMarianTranslator(
                device_choice=settings.model_device,
                model_by_language={
                    "zh": settings.translation_model_zh,
                    "zh-Hans": settings.translation_model_zh,
                    "zh-Hant": settings.translation_model_zh,
                    "ja": settings.translation_model_ja,
                },
            )
            updated, skipped, failed = backfill_english_translations(collection, translator)
            print(
                f"Backfill finished. Translated: {updated}; skipped: {skipped}; "
                f"failed: {failed}."
            )
            return 0

        run_filter = {"window_start": window_start, "window_end": window_end}
        attempt_started_at = datetime.now(timezone.utc)
        collection.database.collection_runs.update_one(run_filter, {"$set": {"status": "running", "attempt_started_at": attempt_started_at, "source_results": []}, "$unset": {"error": ""}}, upsert=True)
        source_results = []
        print(f"Collection window: {window_start.isoformat()} to {window_end.isoformat()}")
        print(f"Loading local summarization model: {settings.model_name}")
        summarizer = LocalMTS5Summarizer(settings.model_name, settings.model_device)
        translator = LocalMarianTranslator(
            device_choice=settings.model_device,
            model_by_language={
                "zh": settings.translation_model_zh,
                "zh-Hans": settings.translation_model_zh,
                "zh-Hant": settings.translation_model_zh,
                "ja": settings.translation_model_ja,
            },
        )
        total_updated = total_skipped = total_failed = 0
        for feed in feeds:
            print(f"[feed] {feed['name']} ({feed['language']})")
            entries_override = None
            if feed.get("source_type") == "mediacloud_api":
                try:
                    entries_override = fetch_mediacloud_entries(
                        feed, window_start, window_end
                    )
                except Exception as error:
                    print(f"[feed error] {feed['name']}: {error}")
                    total_failed += 1
                    source_results.append({"name": feed["name"], "status": "failed", "failed": 1, "error": f"{type(error).__name__}: {str(error)[:250]}"})
                    collection.database.collection_runs.update_one(run_filter, {"$set": {"source_results": source_results}})
                    continue
            updated, skipped, failed = collect_feed(
                feed=feed,
                companies=companies,
                collection=collection,
                summarizer=summarizer,
                translator=translator,
                timeout_seconds=settings.http_timeout_seconds,
                user_agent=settings.user_agent,
                window_start=window_start,
                window_end=window_end,
                topics=topics,
                entries_override=entries_override,
            )
            source_results.append({"name": feed["name"], "status": "partial" if failed else "completed", "saved": updated, "skipped": skipped, "failed": failed})
            collection.database.collection_runs.update_one(run_filter, {"$set": {"source_results": source_results}})
            total_updated += updated
            total_skipped += skipped
            total_failed += failed
        print(
            f"Finished. Saved/updated: {total_updated}; skipped: {total_skipped}; "
            f"failed: {total_failed}."
        )
        dashboard_config = load_yaml(Path("config/dashboard.yaml"))
        prices = fetch_price_snapshots(
            dashboard_config.get("stocks", []), pending_price_start_dates(collection.database)
        )
        brief = make_daily_brief(
            database=collection.database,
            window_start=window_start,
            window_end=window_end,
            timezone_name=settings.window_timezone,
            summarizer=summarizer,
            companies=companies,
            impact_model_name=settings.impact_model_name,
            impact_model_device=settings.impact_model_device,
            analysis_model_name=settings.analysis_model_name,
            analysis_model_device=settings.analysis_model_device,
            prices=prices,
            saved=total_updated,
            skipped=total_skipped,
            failed=total_failed,
        )
        print(
            f"Daily brief saved for {brief['report_date']}: "
            f"{brief['article_count']} articles; impact model {brief['impact_analysis']['status']}."
        )
        collection.database["collection_runs"].update_one(
            {"window_start": window_start, "window_end": window_end},
            {
                "$set": {
                    "window_start": window_start,
                    "window_end": window_end,
                    "timezone": settings.window_timezone,
                    "updated_at": datetime.now(timezone.utc),
                    "status": "partial" if total_failed else "completed",
                    "source_results": source_results,
                    "attempt_started_at": attempt_started_at,
                    "saved": total_updated,
                    "skipped": total_skipped,
                    "failed": total_failed,
                },
                "$setOnInsert": {"started_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
    except Exception as error:
        if "run_filter" in locals():
            collection.database.collection_runs.update_one(run_filter, {"$set": {"status": "failed", "updated_at": datetime.now(timezone.utc), "error": f"{type(error).__name__}: {str(error)[:250]}"}})
        raise
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
