import json

from GPTagger.indexer import Indexer, Tag

cases = json.load(open("tests/test_cases/indexer.json"))


def test_multiple_line():
    case = cases["multiple-line"]
    indexer = Indexer()
    res = indexer.index([case["input"]], case["text"])

    assert len(res) == len(case["output"])

    for i in range(len(res)):
        assert res[i].text == case["output"][i]


def test_unescape_regex():
    case = cases["regex-escape"]
    indexer = Indexer()
    res = indexer.index([case["input"]], case["text"])

    assert len(res) == len(case["output"])

    for i in range(len(res)):
        assert res[i].text == case["output"][i]


def test_bad_first_token():
    case = cases["bad-first-token"]
    indexer = Indexer()
    res = indexer.index([case["input"]], case["text"])

    assert len(res) == len(case["output"])

    for i in range(len(res)):
        assert res[i].text == case["output"][i]


def test_mutiple_matches():
    case = cases["multiple-matches"]
    indexer = Indexer()
    res = indexer.index([case["input"]], case["text"])

    assert len(res) == len(case["output"])

    for i in range(len(res)):
        assert res[i].text == case["output"][i]


def test_overlapping_one():
    indexer = Indexer()

    inputs = [Tag(1, 3, ""), Tag(0, 3, ""), Tag(2, 3, "")]

    res = indexer.resolve_overlap(inputs)

    assert len(res) == 1
    assert res[0] == Tag(2, 3, "")


def test_overlapping_two():
    indexer = Indexer()

    inputs = [
        Tag(1, 3, ""),
        Tag(3, 5, ""),
        Tag(0, 3, ""),
        Tag(4, 5, ""),
    ]

    res = indexer.resolve_overlap(inputs)

    assert len(res) == 2
    assert res[0] == Tag(1, 3, "")
    assert res[1] == Tag(4, 5, "")
