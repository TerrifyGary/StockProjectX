"""Fast executable-app checks: no crawling, checkpoints, or production writes."""
import importlib
import subprocess
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import mongomock
from fastapi.testclient import TestClient
from stock_news import api
from stock_news.config import load_yaml, load_companies, load_feeds, load_topics


class AppSmokeTests(TestCase):
    def test_services_import(self):
        for module in ('stock_news.api', 'stock_news.cli', 'stock_news.scheduler'):
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_cli_help_executes(self):
        result = subprocess.run([sys.executable, '-m', 'stock_news.cli', '--help'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--window-start', result.stdout)

    def test_runtime_configuration_loads(self):
        self.assertTrue(load_companies(Path('config/watchlist.yaml')))
        self.assertTrue(load_feeds(Path('config/sources.yaml')))
        self.assertTrue(load_topics(Path('config/topics.yaml')))
        stocks = load_yaml(Path('config/dashboard.yaml'))['stocks']
        self.assertEqual(len({s['id'] for s in stocks}), len(stocks))
        for stock in stocks:
            self.assertTrue(stock['ticker'])
            self.assertTrue(stock['display_symbol'])

    def test_dashboard_and_audit_routes_execute(self):
        db = mongomock.MongoClient(tz_aware=True).db
        with patch.object(api, 'database', db), patch.object(api, '_get_quotes', return_value=[]):
            client = TestClient(api.app)
            dashboard = client.get('/api/dashboard')
            self.assertEqual(dashboard.status_code, 200, dashboard.text)
            self.assertTrue({'stocks', 'news', 'freshness', 'validation'} <= dashboard.json().keys())
            self.assertEqual(client.get('/api/predictions').status_code, 200)
            self.assertEqual(client.get('/api/predictions?model=bad').status_code, 400)

    def test_health_reports_database_failure(self):
        class Database:
            def command(self, name):
                if name != 'ping':
                    raise AssertionError(name)
                return {'ok': 1}
        with patch.object(api, 'database', Database()):
            self.assertEqual(TestClient(api.app).get('/api/health').json(), {'status': 'ok'})
        with patch.object(api, 'database') as db:
            db.command.side_effect = ConnectionError('test database unavailable')
            self.assertEqual(TestClient(api.app).get('/api/health').status_code, 503)
