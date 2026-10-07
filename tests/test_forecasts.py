from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import exchange_calendars as xcals
import mongomock

from stock_news.daily_briefs import make_daily_brief, validation_summary
from stock_news.forecast_tracking import register_forecasts, register_pending_batches, evaluate_pending_forecasts, pending_price_start_dates, tracking_summary
from stock_news.lstm import predict_next_directions, _report_features, _training_samples
from stock_news.trading_sessions import session_pair, session_close

UTC = timezone.utc
ISSUED = datetime(2026, 7, 3, 0, tzinfo=UTC)


def stock(date='2026-07-02', close=100):
    return {'id': 'tesla', 'symbol': 'TSLA', 'ticker': 'TSLA', 'market': 'US', 'date': date, 'close': close, 'previous_direction': 'down'}


def frozen(issued=ISSUED):
    return {
        'report_date': issued.date().isoformat(), 'created_at': issued, 'prices': [stock()],
        'article_keys': ['story-1'], 'impact_analysis': {'model': 'qwen-test', 'directions': [{'symbol': 'TSLA', 'direction': 'up', 'score': .7}]},
        'lstm': {'predictions': [{'security_id': 'tesla', 'symbol': 'TSLA', 'direction': 'up', 'score': .8, 'probability_up': .8, 'input_features': [[.01,.7]] * 5, 'model_version': 'v2'}]},
    }


class SessionsAndLifecycle(unittest.TestCase):
    def setUp(self):
        self.db = mongomock.MongoClient(tz_aware=True).stock_news

    def test_us_holiday_and_taiwan_session(self):
        self.assertEqual(session_pair(stock(), ISSUED)['target_session'], '2026-07-06')
        tw = {'market': 'Taiwan', 'ticker': '2330.TW'}
        pair = session_pair(tw, datetime(2026, 10, 7, 0, tzinfo=UTC))
        self.assertEqual(pair['reference_session'], '2026-10-06')
        self.assertEqual(pair['target_session'], '2026-10-07')
        # Taiwan National Day in a historical calendar excludes October 10.
        self.assertEqual(session_pair(tw, datetime(2025, 10, 10, 0, tzinfo=UTC))['target_session'], '2025-10-13')

    def test_first_issuance_survives_same_day_and_weekend_reruns(self):
        original = frozen()
        register_forecasts(self.db, original)
        rerun = deepcopy(original)
        rerun['lstm']['predictions'][0]['direction'] = 'down'
        rerun['article_keys'] = ['new-story']
        rerun['created_at'] += timedelta(days=1)
        rerun['report_date'] = '2026-07-04'
        register_forecasts(self.db, rerun)
        self.assertEqual(self.db.prediction_records.count_documents({}), 4)
        row = self.db.prediction_records.find_one({'model': 'lstm'})
        self.assertEqual(row['direction'], 'up')
        self.assertEqual(row['issued_at'], ISSUED)
        self.assertEqual(row['input_article_keys'], ['story-1'])
        self.assertEqual(row['target_session'], '2026-07-06')

    def test_exact_target_recovery_and_idempotence(self):
        register_forecasts(self.db, frozen())
        now = datetime(2026, 7, 8, 0, tzinfo=UTC)
        later = stock('2026-07-07', 90)
        self.assertEqual(evaluate_pending_forecasts(self.db, [later], now=now), 0)
        self.assertEqual(pending_price_start_dates(self.db), {'tesla':'2026-07-02'})
        later['history'] = [{'date':'2026-07-06','close':105,'split':0}]
        self.assertEqual(evaluate_pending_forecasts(self.db, [later], now=now), 4)
        self.assertEqual(evaluate_pending_forecasts(self.db, [later], now=now), 0)
        row = self.db.prediction_records.find_one({'model':'lstm'})
        self.assertTrue(row['correct'])
        self.assertEqual(row['target_close'], 105)
        summary = tracking_summary(self.db)
        self.assertEqual(summary['accuracy_percent'], 100)
        self.assertEqual(summary['models']['previous_direction']['paired_with_lstm']['accuracy_percent'], 0)
        self.assertAlmostEqual(summary['brier_score'], .04)

    def test_unfinished_bar_never_scores(self):
        register_forecasts(self.db, frozen())
        price = stock('2026-07-06', 105)
        self.assertEqual(evaluate_pending_forecasts(self.db, [price], now=datetime(2026,7,6,18,tzinfo=UTC)), 0)

    def test_uncertain_coverage_flat_and_split(self):
        candidate = frozen()
        candidate['lstm']['predictions'][0].update(direction='uncertain', probability_up=.55)
        register_forecasts(self.db, candidate)
        target = stock('2026-07-06',100)
        evaluate_pending_forecasts(self.db, [target], now=datetime(2026,7,7,0,tzinfo=UTC))
        summary = tracking_summary(self.db)
        self.assertEqual(summary['coverage_percent'], 0)
        self.assertEqual(summary['evaluated_calls'], 0)
        self.assertIsNone(summary['brier_score'])  # Flat returns are not binary probability labels.
        self.assertEqual(summary['models']['always_up']['accuracy_percent'], 0)
        self.assertEqual(summary['uncertain_calls'], 1)
        other = mongomock.MongoClient(tz_aware=True).db
        register_forecasts(other, frozen())
        target['split'] = 2
        evaluate_pending_forecasts(other,[target],now=datetime(2026,7,7,0,tzinfo=UTC))
        self.assertEqual(tracking_summary(other)['excluded_calls'],1)

    def test_stale_reference_does_not_issue(self):
        candidate = frozen()
        candidate['prices'][0]['date'] = '2026-07-01'
        register_forecasts(self.db,candidate)
        self.assertEqual(self.db.prediction_records.count_documents({}),0)

    def test_interrupted_registration_recovers_original_calls(self):
        candidate = frozen()
        self.db.forecast_briefs.insert_one(candidate)
        self.assertEqual(pending_price_start_dates(self.db), {'tesla':'2026-07-02'})
        register_pending_batches(self.db)
        register_pending_batches(self.db)
        self.assertEqual(self.db.prediction_records.count_documents({}),4)
        recovered = self.db.forecast_briefs.find_one({})
        self.assertEqual(recovered['created_at'], ISSUED)
        self.assertIn('ledger_registered_at', recovered)

    def test_abstained_probabilities_are_still_scored_for_calibration(self):
        candidate = frozen()
        candidate['lstm']['predictions'][0].update(direction='uncertain', probability_up=.55)
        register_forecasts(self.db, candidate)
        evaluate_pending_forecasts(self.db,[stock('2026-07-06',105)],now=datetime(2026,7,7,0,tzinfo=UTC))
        summary=tracking_summary(self.db)
        self.assertEqual(summary['coverage_percent'],0)
        self.assertEqual(summary['probability_samples'],1)
        self.assertAlmostEqual(summary['brier_score'],.2025)

    def test_legacy_scores_do_not_enter_new_metrics(self):
        self.db.brief_history.insert_one({'report_date':'2026-01-01','evaluation':{'lstm':{'evaluated':100,'correct':100}}})
        summary=validation_summary(self.db)
        self.assertEqual(summary['evaluated_calls'],0)
        self.assertIsNone(summary['accuracy_percent'])

    def test_daily_display_refresh_preserves_forecast_inputs(self):
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return ISSUED.astimezone(tz or UTC)
        args = dict(database=self.db, window_start=ISSUED-timedelta(days=2), window_end=ISSUED,
                    timezone_name='Asia/Taipei', summarizer=None, companies=[], impact_model_name='finbert',
                    impact_model_device='cpu', analysis_model_name='qwen', analysis_model_device='cpu',
                    prices=[stock()], saved=0, skipped=0, failed=0)
        with patch('stock_news.daily_briefs.datetime', Clock):
            make_daily_brief(**args)
            args['prices'] = [stock(close=999)]
            make_daily_brief(**args)
        saved = self.db.forecast_briefs.find_one({})
        self.assertEqual(saved['prices'][0]['close'],100)
        self.assertEqual(self.db.current_brief.find_one({})['prices'][0]['close'],999)
        self.assertEqual(self.db.prediction_records.find_one({})['reference_close'],100)
        self.assertFalse(validation_summary(self.db)['legacy_evaluations_included'])


def training_briefs(count=46):
    calendar = xcals.get_calendar('XNYS', start='2026-01-01', end='2026-12-31')
    sessions = calendar.sessions_in_range('2026-02-01','2026-07-01')[:count]
    rows=[]
    for index, session in enumerate(sessions):
        date=session.strftime('%Y-%m-%d')
        snapshot=stock(date, 100 + (index % 2) * 2 + index * .01)
        issued=session_close(snapshot,date)+timedelta(hours=1)
        rows.append({'report_date':issued.date().isoformat(),'created_at':issued,'prices':[snapshot],
                     'impact_analysis':{'directions':[{'symbol':'TSLA','direction':'up','score':.7}]}})
    return rows


class PerStockTraining(unittest.TestCase):
    def test_duplicate_reports_are_not_training_observations(self):
        rows=training_briefs(29)
        repeated=[deepcopy(row) for row in rows for _ in range(4)]
        for i,row in enumerate(repeated):
            row['created_at']+=timedelta(minutes=i%4)
        result=predict_next_directions(repeated[:-1], repeated[-1])
        self.assertEqual(result['per_security'][0]['observations'],29)
        self.assertEqual(result['predictions'],[])

    def test_samples_not_pooled_across_stocks(self):
        rows=training_briefs(29)
        for row in rows:
            row['prices'] += [{**row['prices'][0],'id':f'stock-{i}','symbol':f'S{i}','ticker':f'S{i}'} for i in range(5)]
        with patch('stock_news.lstm._fit_probability') as train:
            result=predict_next_directions(rows[:-1],rows[-1])
            train.assert_not_called()
        self.assertEqual(len(result['per_security']),6)
        self.assertFalse(result['predictions'])

    def test_chronological_cutoff_and_abstention(self):
        rows=training_briefs()
        future=deepcopy(rows[-1])
        future['created_at']+=timedelta(days=10)
        future['prices'][0]['close']=9999
        with patch('stock_news.lstm._fit_probability',return_value=.55) as fit:
            result=predict_next_directions(rows[:-1]+[future],rows[-1])
        prediction=result['predictions'][0]
        self.assertEqual(prediction['direction'],'uncertain')
        self.assertLessEqual(prediction['training_cutoff'],rows[-1]['created_at'])
        self.assertEqual(prediction['training_samples'],41)
        self.assertEqual(len(fit.call_args.args[0]),41)
        self.assertEqual(prediction['input_features'][-1][1],.7)

    def test_gaps_cannot_become_next_session_training_labels(self):
        rows=training_briefs()
        complete=_report_features(rows)['tesla']
        missing=_report_features(rows[:20]+rows[21:])['tesla']
        full_x,_,_=_training_samples(complete,rows[-1]['created_at'])
        gap_x,_,_=_training_samples(missing,rows[-1]['created_at'])
        self.assertLess(len(gap_x),len(full_x)-1)

    def test_training_failure_keeps_optional_model_unavailable(self):
        rows=training_briefs()
        with patch('stock_news.lstm._fit_probability',side_effect=RuntimeError('model error')):
            result=predict_next_directions(rows[:-1],rows[-1])
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['per_security'][0]['status'],'training_failed')
        self.assertFalse(result['predictions'])

    def test_real_model_fit_with_eligible_history(self):
        import torch
        torch.set_num_threads(1)
        rows=training_briefs()
        result=predict_next_directions(rows[:-1],rows[-1])
        self.assertEqual(result['status'],'available')
        prediction=result['predictions'][0]
        self.assertGreaterEqual(prediction['probability_up'],0)
        self.assertLessEqual(prediction['probability_up'],1)
        self.assertIn(prediction['direction'],{'up','down','uncertain'})


class PipelineRegression(unittest.TestCase):
    def test_qwen_input_serializes_mongo_datetimes(self):
        from stock_news.local_analysis import LocalNewsImpactModel
        model = object.__new__(LocalNewsImpactModel)
        article = {'published_at':ISSUED, 'title_en':'Tesla news', 'display_title':'Tesla news',
                   'display_summary':'A reported event.', 'source':{'name':'publisher'}, 'url':'https://example.com/a',
                   'entities':[{'name':'Tesla','tickers':['TSLA']}], 'topics':[]}
        with patch.object(model, '_generate_json', side_effect=[
            {'cues':[{'article_id':0,'symbols':['TSLA'],'direction':'upside','reason':'Reported support','confidence':.8}]},
            {'investor_note':'Review the evidence.'},
        ]) as generate:
            result=model.analyze([article],[{'name':'Tesla','tickers':['TSLA']}],reference_time=ISSUED)
        self.assertIn(ISSUED.isoformat(),generate.call_args_list[0].args[0])
        self.assertIn('PRE-ANALYSIS EVIDENCE BRIEF',generate.call_args_list[0].args[0])
        self.assertEqual(result['upside'][0]['score'],.8)
        self.assertEqual(result['upside'][0]['evidence_priority_score'],100)

    def test_price_capture_excludes_live_bar_and_uses_raw_closes(self):
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import Mock
        from stock_news.daily_briefs import fetch_price_snapshots
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):
                return datetime(2026,7,6,18,tzinfo=UTC).astimezone(tz or UTC)
        frame=pd.DataFrame({'Close':[90.,100.,105.],'Stock Splits':[0.,0.,0.]},
                           index=pd.to_datetime(['2026-07-01','2026-07-02','2026-07-06']))
        ticker=Mock(); ticker.history.return_value=frame
        with patch.dict('sys.modules',{'yfinance':SimpleNamespace(Ticker=lambda _:ticker)}), patch('stock_news.daily_briefs.datetime',Clock):
            prices=fetch_price_snapshots([{'id':'tesla','display_symbol':'TSLA','ticker':'TSLA','market':'US'}],{'tesla':'2026-05-01'})
        self.assertEqual(prices[0]['date'],'2026-07-02')
        self.assertEqual(prices[0]['previous_direction'],'up')
        self.assertEqual(prices[0]['previous_close'],90)
        self.assertEqual(ticker.history.call_args.kwargs['start'],'2026-05-01')
        self.assertFalse(ticker.history.call_args.kwargs['auto_adjust'])


if __name__ == '__main__':
    unittest.main()
