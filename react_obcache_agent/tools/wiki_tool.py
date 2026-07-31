import re
import time
from typing import List, Optional

import requests


WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_INTERVAL_SECONDS = 0.5
MAX_RETRIES = 5


class WikipediaTool:
    _title_cache = {}
    _extract_cache = {}
    _last_request_time = 0.0

    def __init__(self):
        self.current_title: Optional[str] = None
        self.current_extract: Optional[str] = None
        self.lookup_cursor = {}
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ReActWikipediaDemo/1.0 (local learning script; contact: example@example.com)"
            }
        )

    def reset(self) -> None:
        self.current_title = None
        self.current_extract = None
        self.lookup_cursor = {}

    def search(self, query: str) -> str:
        title = self._search_title(query)

        if title is None:
            return f"Could not find [{query}]."

        extract = self._get_page_extract(title)

        if not extract:
            return f"Found [{title}], but could not get page content."

        self.current_title = title
        self.current_extract = extract
        self.lookup_cursor = {}

        short_extract = self._truncate(extract, max_chars=1400)
        return f"{title}: {short_extract}"

    def lookup(self, keyword: str) -> str:
        if self.current_extract is None:
            return "No current page. Please use search[entity] first."

        sentences = self._split_sentences(self.current_extract)

        matched = [
            sentence for sentence in sentences
            if keyword.lower() in sentence.lower()
        ]

        if not matched:
            return f"Could not find keyword [{keyword}] in current page [{self.current_title}]."

        cursor = self.lookup_cursor.get(keyword.lower(), 0)

        if cursor >= len(matched):
            return f"No more results for keyword [{keyword}] in current page [{self.current_title}]."

        self.lookup_cursor[keyword.lower()] = cursor + 1

        return f"(Result {cursor + 1} / {len(matched)}) {matched[cursor]}"

    def _search_title(self, query: str) -> Optional[str]:
        normalized_query = query.strip().lower()
        if normalized_query in self._title_cache:
            return self._title_cache[normalized_query]

        params = {
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }

        response = self._get_with_retry(params)
        data = response.json()

        titles = data[1] if len(data) > 1 else []
        descriptions = data[2] if len(data) > 2 else []
        for index, candidate in enumerate(titles):
            description = descriptions[index] if index < len(descriptions) else ""
            if self._is_relevant_result(query, candidate, description):
                self._title_cache[normalized_query] = candidate
                return candidate

        title = self._full_text_search_title(query)
        self._title_cache[normalized_query] = title
        return title

    def _full_text_search_title(self, query: str) -> Optional[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
        }

        response = self._get_with_retry(params)
        data = response.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return None

        for result in results:
            title = result.get("title")
            if not title:
                continue
            snippet = re.sub(r"<[^>]+>", " ", result.get("snippet", ""))
            if self._is_relevant_result(query, title, snippet):
                return title

        return None

    def _get_page_extract(self, title: str) -> Optional[str]:
        if title in self._extract_cache:
            return self._extract_cache[title]

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "titles": title,
            "format": "json",
            "redirects": 1,
        }

        response = self._get_with_retry(params)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for _, page in pages.items():
            extract = page.get("extract")
            if extract:
                self._extract_cache[title] = extract
                return extract

        self._extract_cache[title] = None
        return None

    def _get_with_retry(self, params: dict) -> requests.Response:
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            response = self.session.get(WIKI_API_URL, params=params, timeout=20)

            if response.status_code != 429:
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_seconds = int(retry_after)
            else:
                sleep_seconds = min(2 ** attempt, 30)

            print(
                f"Wikipedia API rate limited request. "
                f"Retrying in {sleep_seconds}s ({attempt}/{MAX_RETRIES})."
            )
            time.sleep(sleep_seconds)

        response.raise_for_status()
        return response

    def _wait_for_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - WikipediaTool._last_request_time

        if elapsed < REQUEST_INTERVAL_SECONDS:
            time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)

        WikipediaTool._last_request_time = time.time()

    def _split_sentences(self, text: str) -> List[str]:
        text = text.replace("\n", " ")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _truncate(self, text: str, max_chars: int) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _important_terms(self, query: str) -> List[str]:
        terms = re.findall(r"[a-z0-9]+", query.lower())
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "for",
            "in",
            "is",
            "of",
            "on",
            "the",
            "to",
            "what",
            "who",
        }
        return [term for term in terms if term not in stopwords]

    def _is_relevant_result(self, query: str, title: str, context: str = "") -> bool:
        query_terms = self._important_terms(query)
        if not query_terms:
            return True

        candidate_text = f"{title} {context}".lower()
        matched_terms = sum(term in candidate_text for term in query_terms)
        if len(query_terms) == 1:
            return matched_terms == 1

        required_matches = min(2, len(query_terms))
        return matched_terms >= required_matches
