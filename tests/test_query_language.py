import pytest

from omnisearch.query import PlanError, QuerySyntaxError, parse, plan
from omnisearch.query.ast import And, Between, Compare, Contains, InList, IsNull, Match, Not, Or, Similar, to_text
from omnisearch.schema import AttributeField, Schema, TextField, VectorField

SCHEMA = Schema("products", [
    TextField("text"), VectorField("img", 32, "fake", "image"),
    AttributeField("price", "float"), AttributeField("category", "category"), AttributeField("tags", "tags"),
])


def test_full_statement():
    s = parse("""EXPLAIN ANALYZE SEARCH products
        WHERE MATCH(text, 'red shoes', weight=0.3) AND SIMILAR(img, 0.6*IMAGE('a.jpg') + 0.4*'blue' - 'red')
          AND price BETWEEN 10 AND 80.5 AND category IN ('shoes', 'boots') AND NOT (tags = 'used' OR price IS NULL)
        FUSE WEIGHTED(norm='max') LIMIT 7 WITH (strategy='exact', ef=128);""")
    assert s.explain and s.analyze and s.collection == "products" and s.limit == 7
    assert s.fuse.method == "weighted" and s.fuse.params == {"norm": "max"}
    assert s.options == {"strategy": "exact", "ef": 128}
    items = s.where.items
    assert isinstance(items[0], Match) and items[0].opts == {"weight": 0.3}
    sim = items[1]
    assert isinstance(sim, Similar) and [(i.kind, i.weight) for i in sim.items] == [("image", 0.6), ("text", 0.4), ("text", -1.0)]
    assert isinstance(items[2], Between) and items[2].hi == 80.5
    assert isinstance(items[3], InList) and items[3].values == ["shoes", "boots"]
    assert isinstance(items[4], Not) and isinstance(items[4].item, Or)
    assert isinstance(items[4].item.items[1], IsNull)


def test_precedence_and_escapes():
    s = parse("SEARCH p WHERE a = 1 OR b = 2 AND c = 'it''s'")
    assert isinstance(s.where, Or) and isinstance(s.where.items[1], And)
    assert s.where.items[1].items[1].value == "it's"
    assert to_text(s.where) == "a = 1 OR (b = 2 AND c = 'it''s')"


def test_vector_literal_doc_and_contains():
    s = parse("SEARCH p WHERE SIMILAR(img, VECTOR[0.1, -2, 3e-1]) AND SIMILAR(img, DOC('p1')) AND CONTAINS(text, 'no parking')")
    assert s.where.items[0].items[0].value == [0.1, -2.0, 0.3]
    assert s.where.items[1].items[0].kind == "doc"
    assert isinstance(s.where.items[2], Contains)


@pytest.mark.parametrize("bad", [
    "SEARCH", "SEARCH p WHERE", "SEARCH p LIMIT 0", "SEARCH p WHERE price <", "SEARCH p WHERE MATCH(text)",
    "SEARCH p WHERE x ~ 3", "SEARCH p FUSE MAGIC", "SEARCH p WHERE a NOT = 3", "SEARCH p LIMIT 5 extra",
])
def test_syntax_errors(bad):
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_planner_splits_ranked_and_filters():
    lp = plan(parse("SEARCH products WHERE MATCH(text, 'x') AND price < 5 AND SIMILAR(img, 'y') AND SIMILAR(img, 'z')"), SCHEMA)
    assert [c.kind for c in lp.ranked] == ["keyword", "vector", "vector"]
    assert [c.name for c in lp.ranked] == ["text", "img", "img#2"]
    assert isinstance(lp.filter, Compare)


@pytest.mark.parametrize("bad", [
    "SEARCH products WHERE MATCH(img, 'x')",
    "SEARCH products WHERE SIMILAR(text, 'x')",
    "SEARCH products WHERE nosuch = 1",
    "SEARCH products WHERE category < 3",
    "SEARCH products WHERE price = 'cheap'",
    "SEARCH products WHERE MATCH(text, 'a') OR price < 3",
    "SEARCH products WHERE NOT MATCH(text, 'a')",
    "SEARCH products WHERE price < 3 WITH (strategy='fast')",
    "SEARCH products WHERE price < 3 WITH (bogus=1)",
    "SEARCH products WHERE MATCH(text, 'a') AND SIMILAR(img, 'b') WITH (topk='ta')",
])
def test_plan_errors(bad):
    with pytest.raises(PlanError):
        plan(parse(bad), SCHEMA)
