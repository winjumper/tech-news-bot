import os
import time
import hashlib
import sqlite3
import io
from datetime import datetime, timedelta

import requests
import feedparser
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from PIL import Image


# Load environment variables from a .env file if present.  This allows
# developers to set secrets locally without hard‑coding them in the
# repository.  When running in GitHub Actions, the values will be
# provided via environment variables set in the workflow.
load_dotenv()


DB_PATH = "news.db"
# List of RSS/Atom sources to monitor.  Feel free to add or remove
# feeds here; the script will iterate over each source on every run.
SOURCES = [
    "https://www.gsmarena.com/rss-news-reviews.php3",
    "https://www.androidpolice.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.tomshardware.com/feeds/all",
    "https://www.xda-developers.com/feed/",
    "https://www.engadget.com/rss.xml",
    "https://feeds.arstechnica.com/arstechnica/gadgets/",
    "https://9to5google.com/feed/",
    "https://www.anandtech.com/rss/",
    "https://news.samsung.com/global/feed",
    "https://nvidianews.nvidia.com/rss",
]

# Maximum number of posts to publish on each run.  This can be
# overridden via the MAX_POSTS environment variable.  Setting a
# reasonable limit helps avoid flooding your channels with too many
# consecutive posts and gives the script time to reset between runs.
MAX_POSTS = int(os.getenv("MAX_POSTS", "3"))


def get_db_connection():
    """Create the SQLite database if it doesn't exist and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            published TEXT,
            posted_tg INTEGER DEFAULT 0,
            posted_vk INTEGER DEFAULT 0
        )
        """
    )
    return conn


def hash_id(url: str) -> str:
    """Generate a stable hash for a given URL using SHA-256."""
    return hashlib.sha256(url.encode()).hexdigest()


def within_days(published: str, days: int = 3) -> bool:
    """
    Determine whether a published date string falls within the last
    `days` days.  If parsing fails (e.g. missing or malformed date), the
    function returns True so the entry isn't accidentally filtered out.
    """
    try:
        # feedparser._parse_date returns a tuple suitable for datetime
        dt = datetime(*feedparser._parse_date(published)[:6])
        return dt >= datetime.utcnow() - timedelta(days=days)
    except Exception:
        return True

def fetch_items() -> list[dict]:
    items: list[dict] = []
    for feed_url in SOURCES:
        kept = 0
        try:
            print(f"[feed] fetching: {feed_url}")
            feed = feedparser.parse(feed_url)
            total = len(getattr(feed, "entries", []))
            for entry in feed.entries:
                url = entry.link
                title = entry.title
                summary = getattr(entry, "summary", "")
                published = getattr(entry, "published", "")
                if not within_days(published, days=3):
                    continue
                items.append({"url": url, "title": title, "summary": summary, "published": published})
                kept += 1
            print(f"[feed] done: {feed_url} total={total} kept_last3d={kept}")
        except Exception as exc:
            print(f"[feed] error: {feed_url} -> {exc}")
    print(f"[feed] total_kept_items={len(items)}")
    return items


def upsert_items(conn: sqlite3.Connection, items: list[dict]) -> None:
    """Insert new items into the database or ignore if they already exist."""
    cur = conn.cursor()
    for item in items:
        entry_id = hash_id(item["url"])
        try:
            cur.execute(
                "INSERT INTO posts (id, url, title, published) VALUES (?, ?, ?, ?)",
                (entry_id, item["url"], item["title"], item["published"]),
            )
        except sqlite3.IntegrityError:
            # Entry already exists; skip insertion
            pass
    conn.commit()


def rewrite_with_openrouter_ultra(title: str, summary: str, url: str) -> str:
    """
    Use the OpenRouter API to rewrite a headline and summary into a short
    Russian-language digest.  Requires the OPENROUTER_API_KEY environment
    variable to be set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")
    prompt = (
        f"Ты — редактор техно‑новостей. Сжато, 300–400 знаков.\n"
        f"Структура: короткий лид с эмодзи, 1–2 факта (чип/цена/дата), в конце \"Источник: {url}\".\n"
        f"Без домыслов, только из заголовка/анонса.\n"
        f"Заголовок: {title}\n"
        f"Анонс: {summary[:400]}"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://example.com",
        "X-Title": "Tech Rewriter",
    }
    data = {
        "model": "openrouter/auto",
        "messages": [
            {
                "role": "system",
                "content": "Отвечай по‑русски, сжато и фактологично.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=data,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_og_image(page_url: str) -> str | None:
    """
    Attempt to retrieve the Open Graph image URL from a webpage.  Returns
    None if no OG image is found or if any error occurs.
    """
    try:
        resp = requests.get(
            page_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tag = soup.find("meta", property="og:image") or soup.find(
            "meta", attrs={"name": "og:image"}
        )
        if not tag:
            return None
        img_url = tag.get("content", "").strip()
        if not img_url:
            return None
        # Normalize protocol-relative URLs
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        return img_url
    except Exception:
        return None


def fetch_and_resize(img_url: str, max_w: int = 1280) -> io.BytesIO:
    """
    Download an image and resize it to a maximum width while preserving
    aspect ratio.  Returns an in-memory bytes buffer containing the
    JPEG‑encoded image.
    """
    resp = requests.get(
        img_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    w, h = image.size
    if w > max_w:
        new_h = int(h * (max_w / w))
        image = image.resize((max_w, new_h), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    output.seek(0)
    return output


def post_telegram(text: str, photo: bytes | None = None) -> None:
    """
    Publish a message (and optional photo) to a Telegram channel.  The
    TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables must be
    provided.  If a photo is supplied, it will be uploaded via
    sendPhoto; otherwise sendMessage is used.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials are missing; skipping Telegram post")
        return
    if photo:
        files = {"photo": ("image.jpg", photo, "image/jpeg")}
        data = {"chat_id": chat_id, "caption": text, "parse_mode": "HTML"}
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        resp = requests.post(url, data=data, files=files, timeout=30)
    else:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=30,
        )
    resp.raise_for_status()


def post_vk(text: str, photo: bytes | None = None) -> None:
    """
    Publish a message (and optional photo) to a VK group wall.  The
    VK_GROUP_TOKEN and VK_GROUP_ID environment variables must be set.
    When a photo is supplied, it is first uploaded, then saved, and
    finally attached to the wall post.
    """
    token = os.getenv("VK_GROUP_TOKEN")
    group_id = os.getenv("VK_GROUP_ID")
    if not token or not group_id:
        print("VK credentials are missing; skipping VK post")
        return
    if photo:
        # Step 1: get an upload URL for the group wall
        upload_server_resp = requests.get(
            "https://api.vk.com/method/photos.getWallUploadServer",
            params={
                "group_id": group_id,
                "v": "5.199",
                "access_token": token,
            },
            timeout=20,
        ).json()
        upload_url = upload_server_resp["response"]["upload_url"]
        # Step 2: upload the photo
        upload_resp = requests.post(
            upload_url,
            files={"photo": ("image.jpg", photo, "image/jpeg")},
            timeout=30,
        ).json()
        # Step 3: save the uploaded photo
        save_resp = requests.get(
            "https://api.vk.com/method/photos.saveWallPhoto",
            params={
                "group_id": group_id,
                "photo": upload_resp["photo"],
                "server": upload_resp["server"],
                "hash": upload_resp["hash"],
                "v": "5.199",
                "access_token": token,
            },
            timeout=20,
        ).json()
        saved = save_resp["response"][0]
        attachment = f"photo{saved['owner_id']}_{saved['id']}"
        # Step 4: post to the group wall with the attachment
        post_resp = requests.post(
            "https://api.vk.com/method/wall.post",
            data={
                "owner_id": f"-{group_id}",
                "from_group": 1,
                "message": text,
                "attachments": attachment,
                "v": "5.199",
                "access_token": token,
            },
            timeout=20,
        )
        post_resp.raise_for_status()
    else:
        # Post without an attachment
        post_resp = requests.post(
            "https://api.vk.com/method/wall.post",
            data={
                "owner_id": f"-{group_id}",
                "from_group": 1,
                "message": text,
                "v": "5.199",
                "access_token": token,
            },
            timeout=20,
        )
        post_resp.raise_for_status()


def process_item(item: dict, conn: sqlite3.Connection) -> bool:
    """
    Process a single news item: rewrite it, fetch a photo, post to Telegram
    and VK, and update the database.  Returns True if the post was
    successful and should count against the MAX_POSTS limit.
    """
    url = item["url"]
    title = item["title"]
    summary = item["summary"]
    try:
        text = rewrite_with_openrouter_ultra(title, summary, url)
    except Exception as exc:
        print(f"OpenRouter API failure for {url}: {exc}")
        return False
    # Attempt to get an OG image; if that fails, we'll post without a photo
    photo_bytes: bytes | None = None
    img_url = extract_og_image(url)
    if img_url:
        try:
            buf = fetch_and_resize(img_url)
            photo_bytes = buf.getvalue()
        except Exception as exc:
            print(f"Image processing failure for {url}: {exc}")
            photo_bytes = None
    # Post to Telegram
    if photo_bytes:
        try:
            post_telegram(text, photo_bytes)
        except Exception as exc:
            print(f"Telegram photo post failed for {url}: {exc}")
            post_telegram(text)
    else:
        post_telegram(text)
    # Post to VK
    if photo_bytes:
        try:
            post_vk(text, photo_bytes)
        except Exception as exc:
            print(f"VK photo post failed for {url}: {exc}")
            post_vk(text)
    else:
        post_vk(text)
    # Update the database to mark as posted
    entry_id = hash_id(url)
    cur = conn.cursor()
    cur.execute(
        "UPDATE posts SET posted_tg = 1, posted_vk = 1 WHERE id = ?",
        (entry_id,),
    )
    conn.commit()
    return True


def run() -> None:
    """Main entry point.  Fetch items, update DB, and process unposted entries."""
    conn = get_db_connection()
    items = fetch_items()
    upsert_items(conn, items)
    cur = conn.cursor()
    # Fetch unposted items ordered by newest first
    cur.execute(
        "SELECT id, url, title FROM posts WHERE posted_tg = 0 OR posted_vk = 0 ORDER BY ROWID DESC"
    )
    rows = cur.fetchall()
    count = 0
    for _id, url, title in rows:
        if count >= MAX_POSTS:
            break
        # Retrieve the summary from our current batch for context; if not found,
        # default to an empty string.
        summary = next((it["summary"] for it in items if it["url"] == url), "")
        item = {"url": url, "title": title, "summary": summary}
        if process_item(item, conn):
            count += 1
            # Sleep briefly to avoid hitting rate limits or triggering spam
            time.sleep(2)


if __name__ == "__main__":
    run()
