import re

from typing import List
from fuzzywuzzy import fuzz
from dataclasses import dataclass

from GPTagger.logger import log2file


@dataclass
class Tag:
    start: int
    end: int
    text: str


class Indexer:
    def __init__(
        self,
        token_threshold: int = 80,
        phrase_threshold: int = 80,
    ) -> None:
        """Indexer can find the location of queries in the document

        Args:
            token_threshold (int, optional): first and last token matching threshold. Defaults to 80.
            phrase_threshold (int, optional): query and phrase matching threshold. Defaults to 80.
        """
        self.token_threshold = token_threshold
        self.phrase_threshold = phrase_threshold

    def _find_similar_phrase(self, tokens_q: List[str], tokens_d: List[str]) -> str:
        """Find the most similar phrase in the document given a query using fuzzy

        Args:
            tokens_q (List[str]): list of query tokens
            tokens_d (List[str]): list of document tokens

        Returns:
            str: the most similar phrase
        """
        max_ratio = 0
        similar_phrase = None
        for i in range(len(tokens_d) - len(tokens_q) + 1):
            token_orig_first = tokens_d[i]
            token_orig_last = tokens_d[i + len(tokens_q) - 1]

            # either the first or the last token is matched
            if (
                token_orig_first == tokens_q[0]
                or token_orig_last == tokens_q[-1]
                or fuzz.ratio(token_orig_first, tokens_q[0]) > self.token_threshold
                or fuzz.ratio(token_orig_last, tokens_q[-1]) > self.token_threshold
            ):
                text_q = " ".join(tokens_q)
                text_o = " ".join(tokens_d[i : i + len(tokens_q)])
                ratio = fuzz.ratio(text_o, text_q)
                if ratio >= self.phrase_threshold:
                    if ratio > max_ratio:
                        similar_phrase = text_o
                        max_ratio = ratio

        return similar_phrase

    def _find_phrase_location(self, query: str, phrase: str, doc: str) -> List[Tag]:
        """Find the location of the most similar phrase in the document using regex

        Args:
            query (str): the query text, used for double validation
            phrase (str): the phrase text
            doc (str): the document text

        Returns:
            List[Tag]: list of tag with position and text
        """
        tags = []

        regex = r"\s+".join([re.escape(w) for w in phrase.split()])

        # re.S makes dot can match anything including newline
        for match in re.finditer(regex, doc, flags=re.S):
            # filter out bad matching
            if fuzz.ratio(match.group(), query) > self.phrase_threshold:
                text = match.group()
                tags.append(Tag(match.start(), match.end(), text))

        return tags

    def index(self, queries: List[str], doc: str, fname: str = None) -> List[Tag]:
        """Batch call _index(query)

        Args:
            queries (List[str]): list of query text
            doc (str): document text
            fname (str, optional): document file name, used for logging. Defaults to None.

        Returns:
            List[Tag]: list of tag with positions and text
        """
        tags = []
        tokens_d = doc.split()

        for query in queries:
            tokens_q = query.split()
            phrase = self._find_similar_phrase(tokens_q, tokens_d)
            if phrase:
                tags.extend(self._find_phrase_location(query, phrase, doc))
            else:
                log = {"filter_name": "Indexer", "text": query, "fname": fname}
                log2file.info(log)

        tags = sorted(tags, key=lambda x: x.start)

        return tags

    def resolve_overlap(self, tags: List[Tag], fname: str = None) -> List[Tag]:
        """Giving a list of Tags, resolve overlapping issue among them

        Args:
            tags (List[Tag]): list of tags
            fname (str, optional): document file name, used for logging. Defaults to None.

        Returns:
            List[Tag]: list of tags without overlapping
        """

        if not tags:
            return []

        tags = sorted(tags, key=lambda x: x.end)

        ptr = 0
        tags_wo_overlap = []
        while ptr < len(tags):
            tags_overlap = [tags[ptr]]
            ptr += 1

            # overlapping exists
            while ptr < len(tags) and tags[ptr].start < tags_overlap[0].end:
                tags_overlap.append(tags[ptr])
                ptr += 1

            # by default only save the shortest extraction
            min_len = tags_overlap[0].end - tags_overlap[0].start
            min_index = 0

            for i, tag in enumerate(tags_overlap[1:]):
                if tag.end - tag.start < min_len:
                    # exclude old one
                    min_len, min_index = tag.end - tag.start, i + 1

            # logging
            for i, tag in enumerate(tags_overlap):
                if i == min_index:
                    continue
                log = {
                    "filter_name": "overlapping",
                    "text": tag.text,
                    "fname": fname,
                }
                log2file.info(log)

            # add to no overlap
            tags_wo_overlap.append(tags_overlap[min_index])

        return tags_wo_overlap
