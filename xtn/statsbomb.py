"""Download StatsBomb open data directly from GitHub (raw JSON).

https://github.com/statsbomb/open-data

    data/matches/<competition_id>/<season_id>.json   list of matches
    data/events/<match_id>.json                     list of events of one match

Raw files are cached gzip-compressed in ``<data_dir>/raw`` so they only have
to be downloaded once and can be inspected afterwards.
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _get_json(url: str, tries: int = 5):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=(10, 30))
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _cached(path: str, url: str):
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    data = _get_json(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data


def matches(raw_dir: str, competition_id: int, season_id: int) -> list[dict]:
    path = os.path.join(raw_dir, "matches", f"{competition_id}_{season_id}.json.gz")
    return _cached(path, f"{BASE_URL}/matches/{competition_id}/{season_id}.json")


def events(raw_dir: str, match_id: int) -> list[dict]:
    path = os.path.join(raw_dir, "events", f"{match_id}.json.gz")
    return _cached(path, f"{BASE_URL}/events/{match_id}.json")


def _ensure_events(raw_dir: str, match_id: int) -> None:
    """Download one match's events into the cache without keeping them in memory."""
    path = os.path.join(raw_dir, "events", f"{match_id}.json.gz")
    if not os.path.exists(path):
        events(raw_dir, match_id)


def download_events(raw_dir: str, match_ids, max_workers: int = 6):
    """Download (and cache) the events of many matches in parallel."""
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_ensure_events, raw_dir, m): m for m in match_ids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:  # pragma: no cover
                failed.append(futs[f])
                print(f"  match {futs[f]} failed: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(futs)} matches", flush=True)
    return failed
