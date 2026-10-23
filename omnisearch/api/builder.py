"""Turn the web UI's form (text box, uploaded image, filters) into OmniQL."""
from __future__ import annotations

from omnisearch.encoders.base import encoder_modalities


def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def build_query(coll, text: str | None = None, image_path: str | None = None, filters: dict | None = None,
                fusion: str = "rrf", limit: int = 12, options: dict | None = None) -> str:
    schema = coll.schema
    clauses = []
    text = (text or "").strip()
    if text:
        for tf in schema.text_fields()[:1]:
            clauses.append(f"MATCH({tf.name}, {_q(text)})")
        for vf in schema.vector_fields() + schema.multivector_fields():
            if "text" in encoder_modalities(vf.encoder):
                clauses.append(f"SIMILAR({vf.name}, {_q(text)})")
    if image_path:
        for vf in schema.vector_fields() + schema.multivector_fields():
            if "image" in encoder_modalities(vf.encoder):
                clauses.append(f"SIMILAR({vf.name}, IMAGE({_q(image_path)}))")
    for name, spec in (filters or {}).items():
        typ = schema.get(name).type
        if typ in ("int", "float"):
            lo, hi = spec.get("min"), spec.get("max")
            if lo not in (None, "") and hi not in (None, ""):
                clauses.append(f"{name} BETWEEN {float(lo):g} AND {float(hi):g}")
            elif lo not in (None, ""):
                clauses.append(f"{name} >= {float(lo):g}")
            elif hi not in (None, ""):
                clauses.append(f"{name} <= {float(hi):g}")
        elif spec:
            values = spec if isinstance(spec, list) else [spec]
            clauses.append(f"{name} IN ({', '.join(_q(v) for v in values)})")
    q = f"SEARCH {coll.name}"
    if clauses:
        q += "\nWHERE " + "\n  AND ".join(clauses)
    q += {"rrf": "\nFUSE RRF(k=60)", "weighted": "\nFUSE WEIGHTED(norm='max')", "auto": "\nFUSE AUTO"}[fusion]
    q += f"\nLIMIT {int(limit)}"
    opts = {k: v for k, v in (options or {}).items() if v not in (None, "", "auto")}
    if opts:
        q += "\nWITH (" + ", ".join(f"{k}={_q(v) if isinstance(v, str) else v}" for k, v in opts.items()) + ")"
    return q
