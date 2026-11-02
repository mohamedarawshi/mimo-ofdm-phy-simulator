"""Reproduce the original integrated experiment; run from repository root."""
import os
import runpy
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root / 'results')
runpy.run_path(str(root / 'experiments' / 'mimo_ofdm_full.py'), run_name='__main__')
