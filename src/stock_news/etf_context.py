"""Traceable qualitative ETF context; weights require a dated holdings snapshot."""
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from stock_news.config import load_yaml


def build_etf_context(articles: list[dict[str, Any]], reference_time=None, profiles=None):
    now = reference_time or datetime.now(timezone.utc)
    if profiles is None:
        path = Path('config/etfs.yaml')
        profiles = load_yaml(path).get('etfs', []) if path.exists() else []
    result = []
    for profile in profiles:
        snapshot_date = profile.get('holdings_as_of')
        try:
            day = datetime.fromisoformat(str(snapshot_date)).date()
            age = (now.date() - day).days
            status = 'current' if 0 <= age <= 30 else 'future' if age < 0 else 'stale'
        except (ValueError, TypeError):
            status = 'missing'
        holdings = profile.get('holdings', []) if status == 'current' else []
        if status == 'current' and (not holdings or not profile.get('source_url')):
            status = 'missing'
            holdings = []
        links = []
        seen = set()
        for index, article in enumerate(articles):
            text = ' '.join([str(article.get('title_en') or article.get('title') or ''), str(article.get('display_summary') or article.get('summary', {}).get('text_en') or '')]).casefold()
            event = re.sub(r'[^\w]+', ' ', str(article.get('title_en') or article.get('title') or '').casefold()).strip() or article.get('article_key') or str(index)
            if event in seen:
                continue
            symbols = {str(t).upper() for entity in article.get('entities', []) for t in entity.get('tickers', [])}
            kinds = []
            exposures = []
            if profile['symbol'] in symbols:
                kinds.append('direct')
            for holding in holdings:
                if str(holding['symbol']).upper() in symbols:
                    kinds.append('constituent')
                    exposures.append(holding)
            sectors = profile.get('sectors', [])
            if 'technology' in sectors and any(word in text for word in ('semiconductor', 'technology sector', 'chip export', '半導體', '半导体', '半導体')):
                kinds.append('sector')
            if 'taiwan_equity' in sectors and any(word in text for word in ('taiwan', '台灣', '台湾')):
                kinds.append('sector')
            if article.get('topics'):
                kinds.append('macro')
            if kinds:
                links.append({'article_id': index, 'article_key': article.get('article_key'), 'url': article.get('url'), 'types': sorted(set(kinds)), 'holdings': exposures, 'limitation': 'Candidate relationship only; require a supported transmission mechanism. Sector/macro context does not establish a holding or price direction.'})
                seen.add(event)
        result.append({**profile, 'holdings_as_of': str(snapshot_date) if snapshot_date else None, 'holdings_status': status, 'evidence': links})
    return {'method': 'etf_context_v1', 'profiles': result, 'instruction': 'Do not invent holdings or weights. Missing/stale/future holdings cannot support constituent claims. Exposure weight is not an expected return. Explain direct, constituent, sector, or macro mechanisms; preserve conflicting evidence and abstain when insufficient.'}
