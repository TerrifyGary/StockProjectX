"""Read-only HTTP smoke checks against running services; exits nonzero on failure."""
import argparse
import json
import sys
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://localhost:8000')
    parser.add_argument('--frontend', default='http://localhost:5173')
    args = parser.parse_args()
    try:
        with urlopen(args.api.rstrip('/') + '/api/health', timeout=15) as response:
            if json.load(response).get('status') != 'ok':
                raise ValueError('API/MongoDB health failed')
        print('PASS API and MongoDB health')
        with urlopen(args.frontend.rstrip('/') + '/', timeout=15) as response:
            html = response.read().decode()
            if 'id="app"' not in html or '<script' not in html:
                raise ValueError('Frontend app entry is missing')
        print('PASS frontend HTML entry')
        with urlopen(args.frontend.rstrip('/') + '/api/health', timeout=15) as response:
            if json.load(response).get('status') != 'ok':
                raise ValueError('Frontend API proxy failed')
        print('PASS frontend-to-backend proxy')
        return 0
    except Exception as error:
        print(f'FAIL {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
