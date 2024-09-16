from dataclasses import dataclass


@dataclass
class SearchResult:
    score: float
    content: str
    metadata: dict
