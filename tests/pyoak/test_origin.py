from pathlib import Path

import pytest
from pyoak.origin import (
    NO_ORIGIN,
    NO_POSITION,
    NO_SOURCE,
    SETS_DELIM,
    SOURCE_OPTIMIZED_SERIALIZATION_KEY,
    URI_DELIM,
    CodeOrigin,
    CodePoint,
    CodeRange,
    EntireSourcePosition,
    FileSource,
    MemoryTextSource,
    MultiOrigin,
    NoOrigin,
    NoPosition,
    NoSource,
    Origin,
    Position,
    PositionSet,
    Source,
    SourceSet,
    TextFileSource,
    TextSource,
    XMLFileOrigin,
    XMLPath,
    concat_origins,
    get_code_range,
    merge_origins,
)


@pytest.fixture
def test_file(tmp_path: Path) -> TextFileSource:
    test_file_name = "test_file.txt"
    test_file_path = tmp_path / test_file_name
    test_file_path.write_text(
        """Testing position
123
456
789
    """
    )
    return TextFileSource(test_file_path)


def test_source_set() -> None:
    s = MemoryTextSource("Test")
    s1 = MemoryTextSource("Test1")
    source_set = SourceSet((s, s1))
    assert s in source_set.sources
    assert s1 in source_set.sources
    assert s in source_set
    assert s1 in source_set
    assert len(source_set.sources) == 2
    assert len(source_set) == 2
    assert list(source_set) == [s, s1]
    assert source_set.source_type == "SourceSet"
    assert source_set.get_raw() is None  # type: ignore[func-returns-value]
    assert source_set.fqn == f"SourceSet({s.fqn}{SETS_DELIM}{s1.fqn})"

    source_set = SourceSet.from_json(source_set.to_json())


def test_memory_text_source() -> None:
    s = MemoryTextSource("Test")
    assert s.get_raw() == "Test"
    assert s.fqn == str(id(s))
    assert s.source_type == "<memory>"
    assert s.source_uri == str(id(s))

    # In memory strings won't compare equal, because their source_uri is unique per object
    assert s != MemoryTextSource("Test")

    s = MemoryTextSource("Test", source_uri="test_origin.py")
    assert s.get_raw() == "Test"
    assert s.fqn == "test_origin.py"
    assert s.source_type == "<memory>"
    assert s.source_uri == "test_origin.py"

    serialized = s.to_json()
    # _raw is not serialized
    assert "_raw" not in serialized

    deserialized = MemoryTextSource.from_json(s.to_json())
    assert deserialized is s


def test_file_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    test_file_name = "test_file_source.txt"
    test_file_abs_path = tmp_path / test_file_name
    test_file_rel_path = test_file_abs_path.relative_to(Path.cwd())
    s = FileSource(test_file_rel_path)
    assert s.fqn == test_file_name
    assert s.source_type == "File"
    assert s.source_uri == test_file_name
    assert s.get_raw() is None

    # Serialize and deserialize before loading file
    assert s == FileSource.from_json(s.to_json())

    # test with existing file
    test_file_rel_path.write_bytes(b"Test")
    assert s.get_raw() == b"Test"

    serialized = s.to_json()
    # _raw should not be serialized
    assert "_raw" not in serialized

    deserialized = s.from_json(serialized)
    # But thanks to Source magic it should restore after deserialization
    assert deserialized is s
    # but it should still be equal to another object with same path
    assert deserialized == FileSource(test_file_rel_path)


def test_text_file_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    test_file_name = "test_file_source.txt"
    test_file_abs_path = tmp_path / test_file_name
    test_file_rel_path = test_file_abs_path.relative_to(Path.cwd())
    s = TextFileSource(test_file_rel_path)
    assert s.fqn == test_file_name
    assert s.source_type == "File"
    assert s.source_uri == test_file_name
    assert s.get_raw() is None

    assert s == TextFileSource.from_json(s.to_json())

    # test non-utf8 encoding
    test_file_rel_path.write_text("Test", encoding="windows-1252")
    assert s.get_raw() == "Test"

    # test with reader
    assert (
        TextFileSource(test_file_rel_path).get_raw(
            reader=lambda p: p.read_text(encoding="windows-1252")
        )
        == "Test"
    )

    serialized = s.to_json()
    # _raw should not be serialized
    assert "_raw" not in serialized

    deserialized = s.from_json(serialized)
    # But thanks to Source magic it should restore after deserialization
    assert deserialized is s
    # but it should still be equal to another object with same path
    assert deserialized == TextFileSource(test_file_rel_path)


def test_entire_file_position() -> None:
    p = EntireSourcePosition()
    p1 = EntireSourcePosition()
    assert p is p1
    assert p.fqn == "(entire source)"
    assert str(p) == p.fqn
    assert repr(p) == "EntireSourcePosition()"

    s = MemoryTextSource("Test")
    o = Origin(s, p)
    assert o == Origin.from_json(o.to_json())


def test_position_set() -> None:
    p = EntireSourcePosition()
    p2 = XMLPath("test")

    ps = PositionSet((p, p2))

    assert p in ps.positions
    assert p2 in ps.positions
    assert p in ps
    assert p2 in ps
    assert len(ps.positions) == 2
    assert len(ps) == 2
    assert list(ps) == [p, p2]
    assert ps.fqn == f"PositionSet({p.fqn}{SETS_DELIM}{p2.fqn})"

    assert ps == PositionSet.from_json(ps.to_json())


def test_code_point() -> None:
    with pytest.raises(ValueError):
        # index must be >= 0
        CodePoint(-1, 0, 0)

    with pytest.raises(ValueError):
        # line must be >= 1
        CodePoint(0, 0, 0)

    with pytest.raises(ValueError):
        # column must be >= 0
        CodePoint(0, 1, -1)

    cp1 = CodePoint(0, 1, 0)
    cp1_1 = CodePoint(0, 1, 0)
    cp2 = CodePoint(1, 1, 1)

    assert cp1 < cp2
    assert cp1 <= cp2
    assert cp2 > cp1
    assert cp2 >= cp1
    assert cp1 == cp1_1

    assert CodePoint.from_json(cp1.to_json()) == cp1


def test_code_range() -> None:
    with pytest.raises(ValueError):
        # start must be <= end
        CodeRange(CodePoint(1, 1, 1), CodePoint(0, 1, 0))

    cp1 = CodePoint(0, 1, 0)
    cp2 = CodePoint(10, 2, 5)
    cp2_1 = CodePoint(11, 3, 0)
    cp3 = CodePoint(20, 3, 10)

    cr1 = CodeRange(cp1, cp2)
    cr2 = CodeRange(cp2_1, cp3)
    cr3 = CodeRange(cp1, cp3)

    assert cr1 < cr2
    assert cr1 <= cr2
    assert cr2 > cr1
    assert cr2 >= cr1
    assert cr1 in cr3
    assert cr2 in cr3
    assert cr1 + cr2 == cr3

    assert CodeRange.from_json(cr1.to_json()) == cr1


def test_multi_origin() -> None:
    # Different sources
    p = EntireSourcePosition()
    p2 = XMLPath("test")
    ps = PositionSet((p, p2))

    s = MemoryTextSource("Test")
    s1 = MemoryTextSource("Test1")
    source_set = SourceSet((s, s1))

    o1 = Origin(s, p)
    o2 = Origin(s1, p2)

    mo = MultiOrigin([o1, o2])

    assert p in mo.position
    assert p2 in mo.position
    assert len(mo.position) == 2
    assert isinstance(mo.source, SourceSet)
    assert s in mo.source
    assert s1 in mo.source
    assert len(mo.source) == 2
    assert mo.fqn == f"{source_set.fqn}{URI_DELIM}{ps.fqn}"
    assert mo.get_raw() == [o1.get_raw(), o2.get_raw()]

    assert mo == MultiOrigin.from_json(mo.to_json())

    # Same sources
    o1 = Origin(s, p)
    o2 = Origin(s, p2)

    mo = MultiOrigin([o1, o2])

    assert p in mo.position
    assert p2 in mo.position
    assert len(mo.position) == 2
    assert s == mo.source
    assert mo.fqn == f"{s.fqn}{URI_DELIM}{ps.fqn}"
    assert mo.get_raw() == [o1.get_raw(), o2.get_raw()]

    assert mo == MultiOrigin.from_json(mo.to_json())


def test_xml_file_origin(test_file: TextFileSource) -> None:
    p = XMLPath("/test")
    o = XMLFileOrigin(test_file, p)

    assert o.get_raw() is None  # type: ignore[func-returns-value]
    assert XMLFileOrigin.from_json(o.to_json()) == o


def test_code_origin(test_file: TextFileSource) -> None:
    s = test_file

    o = CodeOrigin(s, CodeRange(CodePoint(17, 2, 2), CodePoint(17, 2, 2)))
    assert o.get_raw() == ""

    o = CodeOrigin(s, CodeRange(CodePoint(18, 2, 2), CodePoint(25, 4, 1)))
    assert o.get_raw() == "23\n456\n"

    o = CodeOrigin(s, CodeRange(CodePoint(18, 2, 2), CodePoint(29, 5, 1)))
    assert o.get_raw(lambda p: p.read_text()) == "23\n456\n789\n"

    Source.clear_registry()
    deserialized = CodeOrigin.from_json(o.to_json())
    assert deserialized.source._raw is None
    assert deserialized == o


def test_merge_origins(test_file: TextFileSource) -> None:
    p = EntireSourcePosition()
    p2 = XMLPath("test")

    s = MemoryTextSource("Test")
    s1 = MemoryTextSource("Test1")

    osub1 = Origin(s, p)
    osub2 = Origin(s1, p2)

    mo = MultiOrigin([osub1, osub2])

    o1 = XMLFileOrigin(test_file, XMLPath("/test"))

    with pytest.raises(ValueError):
        _ = merge_origins("one", "two")  # type: ignore # intentional wrong type

    assert merge_origins(o1) is o1

    new_origin = merge_origins(o1, mo)
    assert isinstance(new_origin, MultiOrigin)
    assert isinstance(new_origin.source, SourceSet)
    assert isinstance(mo.source, SourceSet)
    assert o1.source is new_origin.source[0]
    assert mo.source[0] is new_origin.source[1]
    assert mo.source[1] is new_origin.source[2]
    assert o1.position is new_origin.position[0]
    assert mo.position[0] is new_origin.position[1]
    assert mo.position[1] is new_origin.position[2]

    assert o1 + mo == new_origin


def test_no_origin() -> None:
    ns = NoSource()
    assert ns.get_raw() is None  # type: ignore[func-returns-value]
    assert ns.fqn == "NoSource"
    assert ns is NoSource()
    assert ns is NO_SOURCE
    assert ns.source_registry_id == -1
    assert ns.as_dict() == {}
    assert Source.as_obj({}) is NO_SOURCE
    assert ns is NoSource.from_json(ns.to_json())

    # Test support for pre-1.1.0 serialization format
    assert ns is Source.as_obj(
        {"source_uri": "NoSource", "source_type": "NoSource", "__type": "NoSource"}
    )

    np = NoPosition()
    assert np.fqn == "NoPosition"
    assert np is NoPosition()
    assert np is NO_POSITION
    assert np.as_dict() == {}
    assert Position.as_obj({}) is NO_POSITION
    assert np is NoPosition.from_json(np.to_json())

    # Test support for pre-1.1.0 serialization format
    assert np is Position.as_obj({"__type": "NoPosition"})

    no = NoOrigin()
    assert no.get_raw() is None  # type: ignore[func-returns-value]
    assert no.fqn == "NoOrigin"
    assert no is NoOrigin()
    assert no is NO_ORIGIN
    assert no.as_dict() == {}
    assert Origin.as_obj({}) is NO_ORIGIN
    assert no == NoOrigin.from_json(no.to_json())

    # Test support for pre-1.1.0 serialization format
    assert no is Origin.as_obj(
        {
            "__type": "NoOrigin",
            "source": {"__type": "NoSource", "source_uri": "NoSource", "source_type": "NoSource"},
            "position": {"__type": "NoPosition"},
        }
    )


def test_optimized_source_serialization() -> None:
    s1 = TextSource("test1", "test1", _raw="test1")
    s2 = MemoryTextSource(_raw="test2")

    o1 = CodeOrigin(s1, get_code_range(0, 1, 0, 4, 1, 4))
    o2 = CodeOrigin(s2, get_code_range(0, 1, 0, 4, 1, 4))
    mo = merge_origins(o1, o2)

    assert s1 in Source._sources
    assert s2 in Source._sources
    assert mo.source in Source._sources

    source_idx = Source._sources[mo.source]
    source_dicts = Source.all_as_dict()

    serialized_mo = mo.as_dict(serialization_options={SOURCE_OPTIMIZED_SERIALIZATION_KEY: True})

    assert serialized_mo["source"]["idx"] == source_idx

    # Delete the sources and clear the registry
    Source.clear_registry()
    del s1
    del s2

    # Now it should fail
    with pytest.raises(ValueError):
        Origin.as_obj(serialized_mo)

    # Load the sources back
    Source.load_serialized_sources(source_dicts)

    # Now it should works
    deserialized_mo = Origin.as_obj(serialized_mo)
    assert deserialized_mo == mo


def test_merge_code_origins() -> None:
    test_source1 = MemoryTextSource("Test1")
    test_source2 = MemoryTextSource("Test2")
    p1 = get_code_range(0, 1, 0, 4, 1, 4)
    p2_adjacent_p1_right = get_code_range(4, 1, 4, 5, 2, 0)
    p3 = get_code_range(1, 1, 1, 9, 3, 4)
    p4_adjacent_p3_left = get_code_range(0, 1, 0, 1, 1, 1)
    p5_non_adjacent_p1 = get_code_range(10, 4, 0, 15, 4, 5)
    p6_overlapping_p5 = get_code_range(12, 4, 2, 17, 4, 7)

    o1 = CodeOrigin(test_source1, p1)
    o2_dif_source = CodeOrigin(test_source2, p1)
    assert isinstance(o1 + o2_dif_source, MultiOrigin)

    o5_non_adjacent_p1 = CodeOrigin(test_source1, p5_non_adjacent_p1)
    assert isinstance(o1 + o5_non_adjacent_p1, MultiOrigin)

    o2_adjacent_p1_right = CodeOrigin(test_source1, p2_adjacent_p1_right)
    osum = o1 + o2_adjacent_p1_right
    assert isinstance(osum, CodeOrigin)
    assert osum.source == test_source1
    assert osum.position == get_code_range(0, 1, 0, 5, 2, 0)

    o3 = CodeOrigin(test_source1, p3)
    o4_adjacent_p3_left = CodeOrigin(test_source1, p4_adjacent_p3_left)
    osum = o3 + o4_adjacent_p3_left
    assert isinstance(osum, CodeOrigin)
    assert osum.source == test_source1
    assert osum.position == get_code_range(0, 1, 0, 9, 3, 4)

    o6_overlapping_p5 = CodeOrigin(test_source1, p6_overlapping_p5)
    osum = o5_non_adjacent_p1 + o6_overlapping_p5
    assert isinstance(osum, CodeOrigin)
    assert osum.source == test_source1
    assert osum.position == get_code_range(10, 4, 0, 17, 4, 7)


def test_concat_origins(test_file: TextFileSource) -> None:
    test_source1 = MemoryTextSource("Test1")
    test_source2 = MemoryTextSource("Test2")
    p1 = get_code_range(0, 1, 0, 4, 1, 4)
    p2_adjacent_p1_right = get_code_range(4, 1, 4, 5, 2, 0)
    p2_1_adjacent_p2_right = get_code_range(5, 2, 0, 7, 2, 2)
    p5_non_adjacent_p1 = get_code_range(10, 4, 0, 15, 4, 5)

    o1 = CodeOrigin(test_source1, p1)
    o2_dif_source = CodeOrigin(test_source2, p1)
    assert concat_origins(o1 + o2_dif_source) == merge_origins(o1, o2_dif_source)

    o5_non_adjacent_p1 = CodeOrigin(test_source1, p5_non_adjacent_p1)
    assert concat_origins(o1 + o5_non_adjacent_p1) == merge_origins(o1, o5_non_adjacent_p1)

    o2_adjacent_p1_right = CodeOrigin(test_source1, p2_adjacent_p1_right)
    o2_2_adjacent_p2_right = CodeOrigin(test_source1, p2_1_adjacent_p2_right)
    osum = concat_origins(o1, o2_adjacent_p1_right, o2_2_adjacent_p2_right)
    assert isinstance(osum, CodeOrigin)
    assert osum.source == test_source1
    assert osum.position == get_code_range(0, 1, 0, 7, 2, 2)

    xml_or = XMLFileOrigin(test_file, XMLPath("/test"))
    osum = concat_origins(o1, o2_adjacent_p1_right, xml_or, o2_2_adjacent_p2_right)
    assert isinstance(osum, MultiOrigin)
    assert len(osum.origins) == 3
    assert isinstance(osum.source, SourceSet)
    assert len(osum.source) == 3
    assert xml_or.source in osum.source
    assert o1.source in osum.source
    assert osum.origins[0].source == o1.source
    assert osum.origins[0].position == get_code_range(0, 1, 0, 5, 2, 0)
    assert osum.origins[1] == xml_or
    assert osum.origins[2] == o2_2_adjacent_p2_right
