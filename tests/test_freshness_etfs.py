from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import patch
import mongomock
from fastapi.testclient import TestClient
from stock_news import api
from stock_news.etf_context import build_etf_context

NOW = datetime.now(timezone.utc)

class EtfContextTests(TestCase):
    def profile(self, day=None):
        return dict(symbol='QQQ', benchmark='Nasdaq-100', source_url='https://example.org/holdings', holdings_as_of=day, holdings=[dict(symbol='NVDA', weight_percent=5)], sectors=['technology'])

    def test_dated_constituent_and_mongo_serialization(self):
        article = dict(title_en='Nvidia earnings', entities=[dict(tickers=['NVDA'])])
        result = build_etf_context([article], NOW, [self.profile(NOW.date())])
        self.assertIn('constituent', result['profiles'][0]['evidence'][0]['types'])
        mongomock.MongoClient().db.snapshots.insert_one(result)

    def test_stale_future_missing_holdings_cannot_create_constituent(self):
        for day in (None, NOW.date()-timedelta(days=31), NOW.date()+timedelta(days=1)):
            with self.subTest(day=day):
                result=build_etf_context([dict(title_en='Nvidia earnings',entities=[dict(tickers=['NVDA'])])], NOW, [self.profile(day)])
                self.assertEqual(result['profiles'][0]['evidence'], [])

    def test_direct_and_macro_do_not_need_holdings(self):
        article=dict(title_en='QQQ rates announcement',entities=[dict(tickers=['QQQ'])],topics=['bonds_interest_rates'])
        result=build_etf_context([article], NOW, [self.profile()])
        self.assertEqual(result['profiles'][0]['evidence'][0]['types'], ['direct','macro'])

    def test_multilingual_sector_and_duplicate_titles(self):
        article=dict(title_en='半導體出口限制')
        result=build_etf_context([article,dict(article)], NOW, [self.profile()])
        self.assertEqual(len(result['profiles'][0]['evidence']),1)
        self.assertEqual(result['profiles'][0]['evidence'][0]['types'],['sector'])

class FreshnessTests(TestCase):
    def setUp(self):
        self.db=mongomock.MongoClient(tz_aware=True).db
        self.patcher=patch.object(api,'database',self.db)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_missing_legacy_and_partial(self):
        result=api.collection_freshness(None,None,NOW)
        self.assertEqual(result['collection_status'],'not_collected')
        self.assertIsNone(result['source_results'])
        self.db.collection_runs.insert_one(dict(failed=0,updated_at=NOW-timedelta(days=1)))
        result=api.collection_freshness(dict(failed=2),dict(window_end=NOW),NOW)
        self.assertEqual(result['collection_status'],'partial')
        self.assertTrue(result['brief_current'])
        self.assertIsNotNone(result['last_success'])

    def test_running_and_stuck_not_success(self):
        self.db.collection_runs.insert_one(dict(failed=0,status='failed',updated_at=NOW))
        result=api.collection_freshness(dict(status='running',attempt_started_at=NOW-timedelta(hours=3)),None,NOW)
        self.assertEqual(result['collection_status'],'running_status_unconfirmed')
        self.assertIsNone(result['last_success'])

    def test_dashboard_api_serializes_freshness_without_network(self):
        with patch.object(api,'_get_quotes',return_value=[]):
            response=TestClient(api.app).get('/api/dashboard')
        self.assertEqual(response.status_code,200)
        self.assertIn('next_scheduled_at',response.json()['freshness'])
        self.assertFalse(response.json()['freshness']['brief_current'])
