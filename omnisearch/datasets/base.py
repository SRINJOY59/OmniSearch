"""Common container for retrieval datasets with relevance judgments."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalDataset:
    """A corpus to ingest, queries, and graded relevance (qrels).

    records: ingestable records (with media paths). They must never contain the
             ground-truth captions used as queries - that would leak the answer
             into the keyword index.
    queries: [{"qid": str, "text": str}]
    qrels:   {qid: {record id: relevance}}
    """

    name: str
    modality: str  # image | video | audio | product
    records: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    qrels: dict = field(default_factory=dict)
    root: str = ""

    def subset(self, n_records: int | None = None, n_queries: int | None = None) -> "RetrievalDataset":
        """First n records, and only queries whose relevant docs are all kept."""
        recs = self.records[:n_records] if n_records else self.records
        keep = {r["id"] for r in recs}
        qs = [q for q in self.queries if self.qrels.get(q["qid"]) and set(self.qrels[q["qid"]]) <= keep]
        if n_queries:
            qs = qs[:n_queries]
        return RetrievalDataset(self.name, self.modality, recs, qs,
                                {q["qid"]: self.qrels[q["qid"]] for q in qs}, self.root)

    def summary(self) -> str:
        return f"{self.name}: {len(self.records)} {self.modality} records, {len(self.queries)} queries"
