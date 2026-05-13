#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BEARER_ENV_NAMES = (
    "X_BEARER_TOKEN",
    "TWITTER_BEARER_TOKEN",
    "TWITTER_API_BEARER_TOKEN",
    "X_API_BEARER_TOKEN",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * min(done, total) / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class ProgressReporter:
    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir
        self.progress_log = outdir / "post_fetch_progress.log"
        self.progress_json = outdir / "post_fetch_progress.json"
        self.lock = threading.Lock()

    def update(
        self,
        stage: str,
        done: int,
        total: int,
        detail: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        percent = 100.0 * done / total if total else 0.0
        doc: dict[str, Any] = {
            "time_utc": utc_now(),
            "stage": stage,
            "done": done,
            "total": total,
            "percent": percent,
            "bar": progress_bar(done, total),
            "detail": detail,
        }
        if extra:
            doc.update(extra)

        line = (
            f"{doc['time_utc']} | {stage:<28} "
            f"{doc['bar']} {percent:6.2f}% ({done}/{total}) {detail}"
        )
        with self.lock:
            self.progress_log.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            tmp = self.progress_json.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            tmp.replace(self.progress_json)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, item: dict[str, Any]) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch post/tweet texts for the representative Community Notes sample."
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("analysis/representative_20k_sample_20260511/sample_20k_post_fetch_queue.csv"),
        help="CSV with sample_id,noteId,tweetId.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/representative_20k_sample_20260511/post_fetch"),
        help="Output directory.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        choices=("auto", "api", "syndication", "oembed", "public"),
        default="auto",
        help=(
            "auto: X API v2 if a bearer token is present, otherwise public endpoints. "
            "public: syndication then oEmbed fallback."
        ),
    )
    parser.add_argument(
        "--bearer-env",
        default=",".join(DEFAULT_BEARER_ENV_NAMES),
        help="Comma-separated env var names to search for an X/Twitter API bearer token.",
    )
    parser.add_argument(
        "--preflight-size",
        type=int,
        default=5,
        help="Number of tweet IDs to test before full fetching.",
    )
    parser.add_argument(
        "--continue-if-preflight-fails",
        action="store_true",
        help="Continue full fetching even when no preflight request succeeds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tweet IDs already present as success records in posts.jsonl.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    return parser.parse_args()


def read_queue(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "noteId", "tweetId"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Queue CSV missing required columns: {sorted(missing)}")
        for row in reader:
            tweet_id = str(row.get("tweetId", "")).strip()
            if not tweet_id:
                continue
            rows.append(
                {
                    "sample_id": str(row.get("sample_id", "")).strip(),
                    "noteId": str(row.get("noteId", "")).strip(),
                    "tweetId": tweet_id,
                }
            )
    return rows


def unique_by_tweet(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        tweet_id = row["tweetId"]
        if tweet_id in seen:
            continue
        seen.add(tweet_id)
        unique.append(row)
    return unique


def load_success_ids(posts_jsonl: Path) -> set[str]:
    ids: set[str] = set()
    if not posts_jsonl.exists():
        return ids
    with posts_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "success" and item.get("tweetId"):
                ids.add(str(item["tweetId"]))
    return ids


def find_bearer_token(env_names: str) -> tuple[str | None, str | None]:
    for name in [x.strip() for x in env_names.split(",") if x.strip()]:
        value = os.environ.get(name)
        if value:
            return value.strip(), name
    return None, None


def http_json(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, Any]]:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        status = getattr(resp, "status", 200)
    text = raw.decode("utf-8", errors="replace")
    return status, json.loads(text)


def failure_record(
    row: dict[str, str],
    provider: str,
    error_type: str,
    message: str,
    attempt: int = 0,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "provider": provider,
        "sample_id": row.get("sample_id"),
        "noteId": row.get("noteId"),
        "tweetId": row.get("tweetId"),
        "error_type": error_type,
        "message": message[:1000],
        "attempt": attempt,
        "fetched_at_utc": utc_now(),
    }


def success_record(
    row: dict[str, str],
    provider: str,
    text: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "success",
        "provider": provider,
        "sample_id": row.get("sample_id"),
        "noteId": row.get("noteId"),
        "tweetId": row.get("tweetId"),
        "text": text,
        "lang": raw.get("lang"),
        "created_at": raw.get("created_at"),
        "author_id": raw.get("author_id"),
        "raw": raw,
        "fetched_at_utc": utc_now(),
    }


def fetch_syndication_once(row: dict[str, str], timeout: float) -> dict[str, Any]:
    params = urlencode({"id": row["tweetId"], "lang": "en"})
    url = f"https://cdn.syndication.twimg.com/tweet-result?{params}"
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Referer": "https://platform.twitter.com/",
    }
    try:
        _, data = http_json(url, headers, timeout)
    except HTTPError as e:
        return failure_record(row, "syndication", f"http_{e.code}", str(e))
    except (URLError, TimeoutError, socket.timeout) as e:
        return failure_record(row, "syndication", type(e).__name__, str(e))
    except Exception as e:
        return failure_record(row, "syndication", type(e).__name__, str(e))

    text = str(data.get("text") or data.get("full_text") or "").strip()
    if not text:
        return failure_record(row, "syndication", "missing_text", json.dumps(data)[:1000])
    return success_record(row, "syndication", text, data)


def strip_html_text(value: str) -> str:
    value = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.I)
    value = re.sub(r"<style[\s\S]*?</style>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def fetch_oembed_once(row: dict[str, str], timeout: float) -> dict[str, Any]:
    tweet_url = f"https://twitter.com/i/web/status/{row['tweetId']}"
    params = urlencode({"url": tweet_url, "omit_script": "true", "dnt": "true"})
    url = f"https://publish.twitter.com/oembed?{params}"
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    }
    try:
        _, data = http_json(url, headers, timeout)
    except HTTPError as e:
        return failure_record(row, "oembed", f"http_{e.code}", str(e))
    except (URLError, TimeoutError, socket.timeout) as e:
        return failure_record(row, "oembed", type(e).__name__, str(e))
    except Exception as e:
        return failure_record(row, "oembed", type(e).__name__, str(e))

    text = strip_html_text(str(data.get("html") or ""))
    if not text:
        return failure_record(row, "oembed", "missing_text", json.dumps(data)[:1000])
    raw = {
        "html": data.get("html"),
        "author_name": data.get("author_name"),
        "author_url": data.get("author_url"),
        "url": data.get("url"),
    }
    return success_record(row, "oembed", text, raw)


def fetch_public_once(row: dict[str, str], timeout: float) -> dict[str, Any]:
    first = fetch_syndication_once(row, timeout)
    if first.get("status") == "success":
        return first
    second = fetch_oembed_once(row, timeout)
    if second.get("status") == "success":
        return second
    return {
        "status": "failed",
        "provider": "public",
        "sample_id": row.get("sample_id"),
        "noteId": row.get("noteId"),
        "tweetId": row.get("tweetId"),
        "error_type": "all_public_providers_failed",
        "message": json.dumps(
            {
                "syndication": {
                    "error_type": first.get("error_type"),
                    "message": first.get("message"),
                },
                "oembed": {
                    "error_type": second.get("error_type"),
                    "message": second.get("message"),
                },
            },
            ensure_ascii=False,
        ),
        "attempt": 0,
        "fetched_at_utc": utc_now(),
    }


def fetch_public_with_retries(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for attempt in range(args.max_retries + 1):
        last = fetch_public_once(row, args.timeout)
        last["attempt"] = attempt
        if last.get("status") == "success":
            return last
        if attempt < args.max_retries:
            time.sleep(args.retry_base_sleep * (attempt + 1))
    return last or failure_record(row, "public", "unknown", "unknown failure")


def chunk_rows(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def fetch_api_batch(
    rows: list[dict[str, str]],
    bearer_token: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ids = [row["tweetId"] for row in rows]
    params = urlencode(
        {
            "ids": ",".join(ids),
            "tweet.fields": (
                "created_at,lang,author_id,public_metrics,possibly_sensitive,"
                "conversation_id,referenced_tweets,entities"
            ),
        }
    )
    url = f"https://api.twitter.com/2/tweets?{params}"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
        "User-Agent": "community-note-representative-sample-fetcher/1.0",
    }

    last_error: dict[str, Any] | None = None
    for attempt in range(args.max_retries + 1):
        try:
            _, data = http_json(url, headers, args.timeout)
            by_id = {str(item.get("id")): item for item in data.get("data", [])}
            errors_by_id: dict[str, dict[str, Any]] = {}
            for err in data.get("errors", []) or []:
                resource_id = str(err.get("resource_id") or err.get("value") or "")
                if resource_id:
                    errors_by_id[resource_id] = err

            out: list[dict[str, Any]] = []
            for row in rows:
                tweet_id = row["tweetId"]
                if tweet_id in by_id:
                    raw = by_id[tweet_id]
                    text = str(raw.get("text") or "").strip()
                    if text:
                        out.append(success_record(row, "api_v2", text, raw))
                    else:
                        out.append(failure_record(row, "api_v2", "missing_text", json.dumps(raw), attempt))
                elif tweet_id in errors_by_id:
                    err = errors_by_id[tweet_id]
                    out.append(
                        failure_record(
                            row,
                            "api_v2",
                            str(err.get("title") or err.get("type") or "api_error"),
                            str(err.get("detail") or err),
                            attempt,
                        )
                    )
                else:
                    out.append(failure_record(row, "api_v2", "not_returned", "ID absent from API response", attempt))
            return out
        except HTTPError as e:
            last_error = {"error_type": f"http_{e.code}", "message": str(e)}
            if e.code == 429 and attempt < args.max_retries:
                time.sleep(args.retry_base_sleep * (attempt + 1) * 5)
                continue
        except (URLError, TimeoutError, socket.timeout) as e:
            last_error = {"error_type": type(e).__name__, "message": str(e)}
        except Exception as e:
            last_error = {"error_type": type(e).__name__, "message": str(e)}

        if attempt < args.max_retries:
            time.sleep(args.retry_base_sleep * (attempt + 1))

    assert last_error is not None
    return [
        failure_record(row, "api_v2", last_error["error_type"], last_error["message"], args.max_retries)
        for row in rows
    ]


def run_preflight(
    rows: list[dict[str, str]],
    provider: str,
    bearer_token: str | None,
    args: argparse.Namespace,
    progress: ProgressReporter,
) -> dict[str, Any]:
    sample = rows[: max(0, min(args.preflight_size, len(rows)))]
    results: list[dict[str, Any]] = []
    progress.update("preflight", 0, len(sample), f"provider={provider}")
    if not sample:
        return {"success": 0, "failed": 0, "results": []}

    if provider == "api":
        if not bearer_token:
            results = [failure_record(row, "api_v2", "missing_bearer_token", "No bearer token env var found") for row in sample]
        else:
            results = fetch_api_batch(sample, bearer_token, args)
            progress.update("preflight", len(sample), len(sample), "api batch tested")
    else:
        for idx, row in enumerate(sample, start=1):
            if provider == "syndication":
                result = fetch_syndication_once(row, args.timeout)
            elif provider == "oembed":
                result = fetch_oembed_once(row, args.timeout)
            else:
                result = fetch_public_once(row, args.timeout)
            results.append(result)
            progress.update("preflight", idx, len(sample), f"success={sum(r.get('status') == 'success' for r in results)}")

    return {
        "success": sum(r.get("status") == "success" for r in results),
        "failed": sum(r.get("status") != "success" for r in results),
        "results": results,
    }


def write_csv_summary(posts_jsonl: Path, success_csv: Path, failure_csv: Path) -> tuple[int, int]:
    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    if not posts_jsonl.exists():
        return 0, 0

    with posts_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "success":
                success_rows.append(
                    {
                        "sample_id": item.get("sample_id"),
                        "noteId": item.get("noteId"),
                        "tweetId": item.get("tweetId"),
                        "provider": item.get("provider"),
                        "lang": item.get("lang"),
                        "created_at": item.get("created_at"),
                        "author_id": item.get("author_id"),
                        "text": item.get("text"),
                    }
                )
            else:
                failure_rows.append(
                    {
                        "sample_id": item.get("sample_id"),
                        "noteId": item.get("noteId"),
                        "tweetId": item.get("tweetId"),
                        "provider": item.get("provider"),
                        "error_type": item.get("error_type"),
                        "message": item.get("message"),
                    }
                )

    if success_rows:
        with success_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(success_rows[0].keys()))
            writer.writeheader()
            writer.writerows(success_rows)
    if failure_rows:
        with failure_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failure_rows)
    return len(success_rows), len(failure_rows)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(args.outdir)
    posts_jsonl = args.outdir / "posts.jsonl"
    success_csv = args.outdir / "posts_success.csv"
    failure_csv = args.outdir / "posts_failed.csv"
    preflight_json = args.outdir / "preflight.json"
    metadata_json = args.outdir / "post_fetch_run_metadata.json"
    summary_json = args.outdir / "post_fetch_summary.json"

    rows_all = read_queue(args.queue)
    rows = unique_by_tweet(rows_all)
    skipped_success = 0
    if args.resume:
        done_ids = load_success_ids(posts_jsonl)
        skipped_success = len(done_ids)
        rows = [row for row in rows if row["tweetId"] not in done_ids]

    bearer_token, bearer_env_name = find_bearer_token(args.bearer_env)
    if args.provider == "auto":
        provider = "api" if bearer_token else "public"
    else:
        provider = args.provider

    metadata = {
        "started_at_utc": utc_now(),
        "queue": str(args.queue),
        "outdir": str(args.outdir),
        "input_rows": len(rows_all),
        "unique_tweet_ids": len(unique_by_tweet(rows_all)),
        "remaining_tweet_ids": len(rows),
        "skipped_existing_success": skipped_success,
        "provider_requested": args.provider,
        "provider_resolved": provider,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "timeout": args.timeout,
        "bearer_token_present": bool(bearer_token),
        "bearer_env_name": bearer_env_name if bearer_token else None,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    progress.update("initialize post fetch", 0, max(len(rows), 1), f"provider={provider}; remaining={len(rows)}")
    preflight = run_preflight(rows, provider, bearer_token, args, progress)
    preflight_json.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")

    if preflight["success"] == 0 and not args.continue_if_preflight_fails:
        detail = "preflight failed; no accessible post source"
        summary = {
            **metadata,
            "finished_at_utc": utc_now(),
            "status": "blocked_no_accessible_post_source",
            "preflight_success": preflight["success"],
            "preflight_failed": preflight["failed"],
            "recommendation": (
                "Provide an X/Twitter API bearer token in X_BEARER_TOKEN or enable an authenticated "
                "post source, then rerun with --resume."
            ),
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        progress.update("blocked", 0, max(len(rows), 1), detail, {"status": summary["status"]})
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 2

    writer = JsonlWriter(posts_jsonl)
    counters = {"success": 0, "failed": 0}
    counters_lock = threading.Lock()

    progress.update("fetch posts", 0, len(rows), f"provider={provider}")

    def write_results(results: list[dict[str, Any]]) -> None:
        for item in results:
            writer.write(item)
        with counters_lock:
            counters["success"] += sum(item.get("status") == "success" for item in results)
            counters["failed"] += sum(item.get("status") != "success" for item in results)

    if provider == "api":
        if not bearer_token:
            raise RuntimeError("Provider api requested but no bearer token is available.")
        chunks = chunk_rows(rows, max(1, min(args.batch_size, 100)))
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(fetch_api_batch, chunk, bearer_token, args) for chunk in chunks]
            for fut in as_completed(futures):
                results = fut.result()
                write_results(results)
                done += len(results)
                progress.update(
                    "fetch posts",
                    done,
                    len(rows),
                    f"success={counters['success']} failed={counters['failed']}",
                )
    else:
        fetcher = {
            "public": lambda row: fetch_public_with_retries(row, args),
            "syndication": lambda row: fetch_syndication_once(row, args.timeout),
            "oembed": lambda row: fetch_oembed_once(row, args.timeout),
        }[provider]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(fetcher, row) for row in rows]
            for fut in as_completed(futures):
                result = fut.result()
                write_results([result])
                done += 1
                if done <= 10 or done % 25 == 0 or done == len(rows):
                    progress.update(
                        "fetch posts",
                        done,
                        len(rows),
                        f"success={counters['success']} failed={counters['failed']}",
                    )

    success_count, failure_count = write_csv_summary(posts_jsonl, success_csv, failure_csv)
    summary = {
        **metadata,
        "finished_at_utc": utc_now(),
        "status": "complete",
        "success_records": success_count,
        "failed_records": failure_count,
        "posts_jsonl": str(posts_jsonl),
        "success_csv": str(success_csv),
        "failure_csv": str(failure_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.update(
        "complete",
        len(rows),
        len(rows),
        f"success={success_count} failed={failure_count}",
        {"status": "complete"},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
