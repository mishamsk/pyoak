import pytest
from pyoak.match.helpers import point_at_index


@pytest.mark.parametrize(
    "input_string,index,length,expected",
    [
        ("a", 0, None, "a\n^"),
        ("a\n", 0, None, "a\n^\n"),
        ("0123456", 3, 2, "0123456\n---^^  "),
        ("0123\n56789\nafter", 7, 4, "0123\n56789\n--^^^\nafter"),
        ("0123\npre56789suf\nafter", 10, 4, "0123\npre56789suf\n  ---^^^^  \nafter"),
    ],
)
def test_point_at_index(input_string: str, index: int, length: int | None, expected: str) -> None:
    if length is None:
        assert point_at_index(input_string, index) == expected
    else:
        assert point_at_index(input_string, index, length) == expected
