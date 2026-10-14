"""Validate the CI dependency gate and live-check failure handling."""
import importlib.util
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch
from urllib.error import URLError
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('smoke_running', ROOT / 'scripts/smoke_running.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeGateTests(TestCase):
    def test_ci_followups_require_successful_smoke_job(self):
        workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
        for name in ('backend-tests', 'frontend-build'):
            job = workflow['jobs'][name]
            self.assertEqual(job['needs'], 'smoke-tests')
            self.assertNotIn('if', job)  # Default success-only dependency behavior.
            self.assertNotIn('continue-on-error', job)
        self.assertNotIn('continue-on-error', workflow['jobs']['smoke-tests'])

    def test_live_failure_returns_nonzero(self):
        with patch.object(sys, 'argv', ['smoke_running.py']), patch.object(smoke, 'urlopen', side_effect=URLError('controlled failure')), patch('sys.stderr', new_callable=StringIO) as output:
            self.assertEqual(smoke.main(), 1)
            self.assertIn('FAIL', output.getvalue())

    def test_nonzero_exit_blocks_followup_command(self):
        with tempfile.TemporaryDirectory() as directory:
            followup = Path(directory) / 'followup.txt'
            # Port 0 is deliberately unusable; no service is stopped or modified.
            check = subprocess.run([sys.executable, str(ROOT / 'scripts/smoke_running.py'), '--api', 'http://127.0.0.1:0'], capture_output=True, text=True, timeout=20)
            if check.returncode == 0:
                followup.write_text('executed')
            self.assertEqual(check.returncode, 1, check.stdout + check.stderr)
            self.assertFalse(followup.exists())
