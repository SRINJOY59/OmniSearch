"""OmniSearch: a cost-based multimodal hybrid (keyword + vector) search engine."""
from omnisearch.engine import Collection, Engine
from omnisearch.schema import (
    AttributeField, MediaField, MultiVectorField, Schema, TextField, VectorField,
)

__version__ = "0.1.0"
__all__ = [
    "Collection", "Engine", "Schema", "MediaField", "TextField", "VectorField",
    "MultiVectorField", "AttributeField",
]
