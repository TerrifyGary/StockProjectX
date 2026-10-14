"""Conservative headline grouping with traceable, preserved publisher reports."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import re
import unicodedata

STOP = {'the','a','an','and','of','to','in','for','on','at','with','as','its','is','by','from'}
CONFLICT = ({'rise','rises','up','increase','increases','gain','gains'}, {'fall','falls','down','decrease','decreases','drop','drops'})


def _title(article):
    return unicodedata.normalize('NFKC', str(article.get('title_en') or article.get('display_title') or article.get('title') or '')).casefold()


def _tokens(article):
    return set(re.findall(r'\w+', _title(article))) - STOP


def _time(article):
    value = article.get('published_at')
    try:
        value = datetime.fromisoformat(value.replace('Z','+00:00')) if isinstance(value,str) else value
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    except (ValueError, AttributeError):
        return None


def _matches(left, right):
    if left.get('url') and left.get('url') == right.get('url'):
        return True
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    ta, tb = _time(left), _time(right)
    if not ta or not tb or abs((ta-tb).total_seconds()) > 48*3600:
        return False
    # Numbers, negation, and opposing direction words can distinguish event updates.
    if set(re.findall(r'\d+(?:\.\d+)?',_title(left))) != set(re.findall(r'\d+(?:\.\d+)?',_title(right))):
        return False
    if bool(a & {'not','no','denies','denied'}) != bool(b & {'not','no','denies','denied'}):
        return False
    if any(a & x and b & y or a & y and b & x for x,y in [CONFLICT]):
        return False
    if a == b:
        return True
    entities_a = {e.get('id') or e.get('name') for e in left.get('entities',[])}
    entities_b = {e.get('id') or e.get('name') for e in right.get('entities',[])}
    return bool(entities_a & entities_b) and len(a & b) >= 4 and len(a & b)/len(a | b) >= .72


def group_stories(articles):
    groups = []
    for article in articles:
        group = next((g for g in groups if _matches(g[0], article)), None)
        if group is None:
            groups.append([article])
        else:
            group.append(article)
    output = []
    for members in groups:
        representative = deepcopy(members[0])
        keys = sorted({a.get('article_key') or a.get('url') or _title(a) for a in members})
        representative['event_id'] = sha256('|'.join(keys).encode()).hexdigest()
        representative['article_keys'] = sorted({a['article_key'] for a in members if a.get('article_key')})
        representative['report_count'] = len(members)
        representative['grouping_method'] = 'conservative_headline_v1'
        representative['sources'] = [{'name':a.get('source',{}).get('name') or 'Unknown publisher', 'url':a.get('url'), 'article_key':a.get('article_key'), 'title':a.get('title_en') or a.get('title'), 'summary':a.get('display_summary') or a.get('summary',{}).get('text_en') or a.get('summary',{}).get('text'), 'published_at':a.get('published_at')} for a in members]
        representative['source_count'] = len({s['name'] for s in representative['sources']})
        representative['entities'] = list({e.get('id') or e.get('name'):e for a in members for e in a.get('entities',[])}.values())
        representative['topics'] = sorted({t for a in members for t in a.get('topics',[])})
        output.append(representative)
    return output
