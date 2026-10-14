from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest import TestCase
from stock_news.event_grouping import group_stories

class EventGroupingTests(TestCase):
    def article(self, title, key='a', day=0):
        return dict(article_key=key,url='https://example.org/'+key,title_en=title,published_at=datetime(2026,10,14,tzinfo=timezone.utc)+timedelta(days=day),entities=[dict(id='tesla',name='Tesla',tickers=['TSLA'])],source=dict(name=key),summary=dict(text_en='Report '+key))

    def test_multisource_group_preserves_originals(self):
        stories=[self.article('Tesla announces new electric vehicle factory'),self.article('Tesla announces a new electric vehicle factory','b')]
        original=deepcopy(stories)
        events=group_stories(stories)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['article_keys'],['a','b'])
        self.assertEqual(events[0]['source_count'],2)
        self.assertEqual(stories,original)
        self.assertEqual(len(events[0]['sources']),2)

    def test_conflicting_updates_numbers_and_dates_stay_separate(self):
        for title, day in [('Tesla earnings rise 20 percent',0),('Tesla earnings fall 10 percent',0),('Tesla earnings rise 10 percent',3)]:
            with self.subTest(title=title,day=day):
                self.assertEqual(len(group_stories([self.article('Tesla earnings rise 10 percent'),self.article(title,'b',day)])),2)

    def test_missing_titles_do_not_merge(self):
        self.assertEqual(len(group_stories([self.article(''),self.article('','b')])),2)

    def test_translation_grouping(self):
        a=self.article('Tesla opens new factory');a['title']='特斯拉開設新工廠'
        b=self.article('Tesla opens a new factory','b');b['title']='テスラが工場を開設'
        self.assertEqual(len(group_stories([a,b])),1)
