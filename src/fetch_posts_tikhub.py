#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://api.tikhub.io/api/v1/twitter/web/fetch_tweet_detail"


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
        self.progress_log = outdir / "tikhub_fetch_progress.log"
        self.progress_json = outdir / "tikhub_fetch_progress.json"
        self.lock = threading.Lock()

    def update(
        self,
        stage: str,
        done: int,
        total: int,
        detail: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        pct = 100.0 * done / total if total else 0.0
        doc: dict[str, Any] = {
            "time_utc": utc_now(),
            "stage": stage,
            "done": done,
            "total": total,
            "percent": pct,
            "bar": progress_bar(done, total),
            "detail": detail,
        }
        if extra:
            doc.update(extra)
        line = f"{doc['time_utc']} | {stage:<28} {doc['bar']} {pct:6.2f}% ({done}/{total}) {detail}"
        with self.lock:
            self.outdir.mkdir(parents=True, exist_ok=True)
            with self.progress_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            tmp = self.progress_json.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
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


class RateLimiter:
    def __init__(self, rate_per_second: float) -> None:
        self.interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.monotonic()
            self.next_time = now + self.interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch tweet/post text through TikHub API.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("analysis/representative_20k_sample_20260511/sample_20k_post_fetch_queue.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/representative_20k_sample_20260511/post_fetch_tikhub"),
    )
    parser.add_argument("--token-file", type=Path, default=Path("secrets/tikhub_token.txt"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rate-limit", type=float, default=8.0, help="Global request rate limit per second.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--request-param", default="tweet_id", help="Query parameter name for tweet id.")
    return parser.parse_args()


def read_token(path: Path) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Token file is empty: {path}")
    return token


def read_queue(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
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


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["tweetId"] in seen:
            continue
        seen.add(row["tweetId"])
        out.append(row)
    return out


def load_success_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "success" and item.get("tweetId"):
                ids.add(str(item["tweetId"]))
    return ids


def collect_keys(obj: Any, prefix: str = "", max_depth: int = 4) -> list[str]:
    if max_depth < 0:
        return []
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name)
            keys.extend(collect_keys(value, name, max_depth - 1))
    elif isinstance(obj, list) and obj:
        keys.extend(collect_keys(obj[0], f"{prefix}[]", max_depth - 1))
    return keys


def find_text_value(obj: Any) -> str | None:
    candidate_keys = (
        "full_text",
        "text",
        "tweet_text",
        "desc",
        "description",
        "content",
        "raw_text",
    )
    if isinstance(obj, dict):
        for key in candidate_keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in obj.values():
            found = find_text_value(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_text_value(value)
            if found:
                return found
    return None


def find_created_at(obj: Any) -> str | None:
    candidate_keys = ("created_at", "createdAt", "create_time", "date", "time")
    if isinstance(obj, dict):
        for key in candidate_keys:
            value = obj.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value)
        for value in obj.values():
            found = find_created_at(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_created_at(value)
            if found:
                return found
    return None


def find_author(obj: Any) -> str | None:
    candidate_keys = ("screen_name", "username", "user_name", "name", "author_name")
    if isinstance(obj, dict):
        user_obj = obj.get("user") or obj.get("author") or obj.get("core")
        if isinstance(user_obj, dict):
            for key in candidate_keys:
                value = user_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in candidate_keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in obj.values():
            found = find_author(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_author(value)
            if found:
                return found
    return None


def request_tikhub(
    row: dict[str, str],
    token: str,
    args: argparse.Namespace,
    limiter: RateLimiter,
) -> dict[str, Any]:
    tweet_id = row["tweetId"]
    params = urlencode({args.request_param: tweet_id})
    url = f"{args.endpoint}?{params}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "community-notes-research-fetcher/1.0",
    }

    last_error: dict[str, str] | None = None
    for attempt in range(args.max_retries + 1):
        limiter.wait()
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=args.timeout) as resp:
                status_code = getattr(resp, "status", 200)
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            text = find_text_value(data)
            if not text:
                return {
                    "status": "failed",
                    "provider": "tikhub",
                    **row,
                    "error_type": "missing_text",
                    "message": "No text-like field found in response.",
                    "http_status": status_code,
                    "response_keys": collect_keys(data)[:300],
                    "raw": data,
                    "attempt": attempt,
                    "fetched_at_utc": utc_now(),
                }
            return {
                "status": "success",
                "provider": "tikhub",
                **row,
                "text": text,
                "created_at": find_created_at(data),
                "author": find_author(data),
                "http_status": status_code,
                "raw": data,
                "attempt": attempt,
                "fetched_at_utc": utc_now(),
            }
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            last_error = {"error_type": f"http_{e.code}", "message": body or str(e)}
            if e.code in {401, 403, 402}:
                break
            if e.code == 429 or e.code >= 500:
                time.sleep(args.retry_base_sleep * (attempt + 1) + random.uniform(0, 0.5))
                continue
            break
        except (URLError, TimeoutError) as e:
            last_error = {"error_type": type(e).__name__, "message": str(e)}
            time.sleep(args.retry_base_sleep * (attempt + 1) + random.uniform(0, 0.5))
        except Exception as e:
            last_error = {"error_type": type(e).__name__, "message": str(e)}
            time.sleep(args.retry_base_sleep * (attempt + 1) + random.uniform(0, 0.5))

    last_error = last_error or {"error_type": "unknown", "message": "unknown error"}
    return {
        "status": "failed",
        "provider": "tikhub",
        **row,
        "error_type": last_error["error_type"],
        "message": last_error["message"][:2000],
        "attempt": args.max_retries,
        "fetched_at_utc": utc_now(),
    }


def write_csv_outputs(posts_jsonl: Path, success_csv: Path, failed_csv: Path) -> tuple[int, int]:
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
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
                        "text": item.get("text"),
                        "created_at": item.get("created_at"),
                        "author": item.get("author"),
                        "fetched_at_utc": item.get("fetched_at_utc"),
                    }
                )
            else:
                failed_rows.append(
                    {
                        "sample_id": item.get("sample_id"),
                        "noteId": item.get("noteId"),
                        "tweetId": item.get("tweetId"),
                        "error_type": item.get("error_type"),
                        "message": item.get("message"),
                        "fetched_at_utc": item.get("fetched_at_utc"),
                    }
                )

    if success_rows:
        with success_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(success_rows[0].keys()))
            writer.writeheader()
            writer.writerows(success_rows)
    if failed_rows:
        with failed_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failed_rows)
    return len(success_rows), len(failed_rows)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    token = read_token(args.token_file)
    posts_jsonl = args.outdir / "posts_tikhub.jsonl"
    success_csv = args.outdir / "posts_success.csv"
    failed_csv = args.outdir / "posts_failed.csv"
    summary_json = args.outdir / "summary.json"
    writer = JsonlWriter(posts_jsonl)
    progress = ProgressReporter(args.outdir)
    limiter = RateLimiter(args.rate_limit)

    all_rows = unique_rows(read_queue(args.queue))
    done_ids = load_success_ids(posts_jsonl) if args.resume else set()
    rows = [row for row in all_rows if row["tweetId"] not in done_ids]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    metadata = {
        "started_at_utc": utc_now(),
        "queue": str(args.queue),
        "outdir": str(args.outdir),
        "endpoint": args.endpoint,
        "request_param": args.request_param,
        "input_unique_tweets": len(all_rows),
        "already_done_success": len(done_ids),
        "remaining_this_run": len(rows),
        "workers": args.workers,
        "rate_limit": args.rate_limit,
        "token_file": str(args.token_file),
    }
    (args.outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.update("initialize", 0, max(len(rows), 1), f"remaining={len(rows)} already_done={len(done_ids)}")

    success = 0
    failed = 0
    done = 0
    lock = threading.Lock()

    def fetch_and_write(row: dict[str, str]) -> dict[str, Any]:
        result = request_tikhub(row, token, args, limiter)
        writer.write(result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_and_write, row) for row in rows]
        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                done += 1  # type: ignore[has-type]
                if result.get("status") == "success":
                    success += 1  # type: ignore[has-type]
                else:
                    failed += 1  # type: ignore[has-type]
                if done <= 10 or done % 25 == 0 or done == len(rows):
                    progress.update(
                        "fetch posts",
                        done,
                        len(rows),
                        f"success_run={success} failed_run={failed} last={result.get('tweetId')}",
                    )

    success_total, failed_total = write_csv_outputs(posts_jsonl, success_csv, failed_csv)
    summary = {
        **metadata,
        "finished_at_utc": utc_now(),
        "status": "complete",
        "success_total": success_total,
        "failed_total": failed_total,
        "posts_jsonl": str(posts_jsonl),
        "success_csv": str(success_csv),
        "failed_csv": str(failed_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.update("complete", done, max(len(rows), 1), f"success_total={success_total} failed_total={failed_total}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
