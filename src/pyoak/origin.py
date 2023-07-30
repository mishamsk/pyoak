from __future__ import annotations

import logging
import typing as t
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .file import read_text_unknown_encoding
from .serialize import DataClassSerializeMixin

if t.TYPE_CHECKING:
    from mashumaro.dialect import Dialect

logger = logging.getLogger(__name__)

URI_DELIM = "::"
SETS_DELIM = "||"

# -----------------------------------------------------------------------------
# ----------------------------- Base Classess ---------------------------------
#
# All source, position & origin types should be derived from these classes
#
# -----------------------------------------------------------------------------


class FQN(ABC):
    @property
    @abstractmethod
    def fqn(self) -> str:
        ...


# -----------------------------------------------------------------------------
# --------------------- Source bases & constants ---------------------------
SourceType = t.TypeVar("SourceType", bound="Source")

SOURCE_OPTIMIZED_SERIALIZATION_KEY = "source_optimized_serialization"
"""Use with (de)serialation functions via `serialization_options` to enable/disable optimized
serialization of sources.

When enabled, source serializaes as a single integer, which is the index
of the source in the registry of all sources.

Be aware that in order to use this, you need to separately serialize all sources
without optimization and before deserialization of objects using sources,
deserialize these and call `Source.load_serialized_sources`

"""


@dataclass(frozen=True)
class Source(DataClassSerializeMixin, FQN):
    """Base class for all source types.

    Provides serialization/deserialization of subclasses.
    Subclasses must be decorated with @dataclass.
    Fields that should not be serialized must start with an underscore.

    Args:
        source_type: The type of the source. If None, the class name is used.
        source_uri: The URI of the source. If None, the fqn is used.
        _raw: The raw source data. If None, some subclasses provide automatic loading.

    """

    source_uri: str
    source_type: str
    _raw: t.Any | None = field(default=None, compare=False, repr=False, kw_only=True)

    # Holds a reference to all instances of this class, used for optimized serialization
    _sources: t.ClassVar[dict[Source, int]] = {}
    _source_idx_to_source: t.ClassVar[dict[int, Source]] = {}

    def __post_init__(self) -> None:
        if self in Source._sources:
            logger.debug(f"Source {self} already exists in registry. Skipping.")
        else:
            Source._sources[self] = len(Source._sources)
            Source._source_idx_to_source[len(Source._sources) - 1] = self

    def __post_serialize__(self, d: dict[str, t.Any]) -> dict[str, t.Any]:
        d = super().__post_serialize__(d)
        d.pop("_raw", None)
        return d

    def _serialize(self) -> dict[str, t.Any]:
        if self._get_serialization_options().get(SOURCE_OPTIMIZED_SERIALIZATION_KEY, False):
            return {"idx": Source._sources[self]}
        return super()._serialize()

    @classmethod
    def _deserialize(cls: t.Type[SourceType], data: dict[str, t.Any]) -> SourceType:
        if cls._get_deserialization_options().get(SOURCE_OPTIMIZED_SERIALIZATION_KEY, False):
            idx = data.get("idx")
            if idx is None:
                raise ValueError("Missing idx in serialized source data")
            ret: SourceType | None = None
            for source, source_idx in Source._sources.items():
                if source_idx == idx:
                    return t.cast(SourceType, source)

            if ret is None:
                raise ValueError(
                    f"Source with idx {idx} not found in registry. Did you forget to load sources?"
                )
        # This may return not the object in the registry, but a new instance
        # if the same source was previously in the registry
        obj = super()._deserialize(data)

        return t.cast(SourceType, Source._source_idx_to_source[Source._sources[obj]])

    @staticmethod
    def load_serialized_sources(sources: list[dict[str, t.Any]]) -> None:
        """Load serialized sources.

        Args:
            data: The list of dictionaries containing the serialized sources.

        """
        for source_dict in sources:
            # This will load sources to the registry (in __post_init__)
            Source.as_obj(source_dict)

    @staticmethod
    def all_as_dict(
        mashumaro_dialect: t.Type[Dialect] | None = None,
    ) -> list[dict[str, t.Any]]:
        """Serialize all registred sources.

        Args:
            mashumaro_dialect: The mashumaro dialect to use for serialization.

        Returns:
            The list of dictionaries containing the serialized sources.

        """

        return [
            source.as_dict(mashumaro_dialect=mashumaro_dialect) for source in Source._sources.keys()
        ]

    @classmethod
    def clear_registry(cls) -> None:
        cls._sources = {}
        cls._source_idx_to_source = {}

    @classmethod
    def list_registered_sources(cls, exclude_no_source: bool = False) -> list[Source]:
        ret = list(cls._sources.keys())
        if exclude_no_source and NoSource() in ret:
            ret.remove(NoSource())
        return ret

    def __str__(self) -> str:
        return f"{self.source_type}@{self.source_uri}"

    @property
    def fqn(self) -> str:
        return self.source_uri

    def get_raw(self) -> t.Any | None:
        return self._raw

    @property
    def source_registry_id(self) -> int:
        return Source._sources[self]


@dataclass(frozen=True)
class TextSource(Source):
    _raw: str | None = field(default=None, compare=False, repr=False, kw_only=True)

    def get_raw(self) -> str | None:
        return self._raw


class Position(DataClassSerializeMixin, FQN):
    """Base class for all position types."""

    pass


@dataclass(frozen=True)
class Origin(DataClassSerializeMixin, FQN):
    """Base class for all origin types."""

    source: Source
    position: Position

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source):
            raise TypeError(f"Expected Source, got {type(self.source)}")

        if not isinstance(self.position, Position):
            raise TypeError(f"Expected Position, got {type(self.position)}")

    def __str__(self) -> str:
        return f"{self.source.source_type}@{self.fqn}"

    @property
    def fqn(self) -> str:
        return f"{self.source.fqn}{URI_DELIM}{self.position.fqn}"

    def get_raw(self) -> t.Any | None:
        return None

    def __add__(self, other: Origin) -> Origin:
        return merge_origins(self, other)


# -----------------------------------------------------------------------------
# --------------------- No Origin Singelton Classes ---------------------------
@dataclass(frozen=True)
class NoSource(Source):
    source_uri: str = field(default="NoSource", init=False)
    source_type: str = field(default="NoSource", init=False)
    _raw: None = field(default=None, compare=False, repr=False, init=False)

    _instance: t.ClassVar[NoSource | None] = None

    def __new__(cls, *args: t.Any, **kwargs: t.Any) -> NoSource:
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def get_raw(self) -> None:
        return None


@dataclass(frozen=True)
class NoPosition(Position):
    _instance: t.ClassVar[NoPosition | None] = None

    def __new__(cls, *args: t.Any, **kwargs: t.Any) -> NoPosition:
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    @property
    def fqn(self) -> str:
        return "NoPosition"


@dataclass(frozen=True)
class NoOrigin(Origin):
    source: NoSource = field(default_factory=NoSource, init=False)
    position: NoPosition = field(default_factory=NoPosition, init=False)

    _instance: t.ClassVar[NoOrigin | None] = None

    def __new__(cls, *args: t.Any, **kwargs: t.Any) -> NoOrigin:
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    @property
    def fqn(self) -> str:
        return "NoOrigin"

    def get_raw(self) -> None:
        return None


# -----------------------------------------------------------------------------
# ----------------------------- Source Classess ---------------------------------


@dataclass(frozen=True)
class SourceSet(Source):
    sources: tuple[Source, ...]
    # Everything else is set automatically
    source_uri: str = field(default="UNSET", init=False)
    source_type: str = field(default="SourceSet", init=False)
    _raw: None = field(default=None, compare=False, repr=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_uri", self.fqn)
        super().__post_init__()

    @property
    def fqn(self) -> str:
        return f"SourceSet({SETS_DELIM.join([s.fqn for s in self.sources])})"

    def get_raw(self) -> None:
        return None

    def __contains__(self, item: Source) -> bool:
        return item in self.sources

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self) -> t.Iterator[Source]:
        return iter(self.sources)

    def __getitem__(self, key: int) -> Source:
        return self.sources[key]


@dataclass(frozen=True)
class MemoryTextSource(TextSource):
    """Represents a text in memory source."""

    _raw: str | None = field(default=None, repr=False, compare=False)
    source_uri: str = field(default="UNSET", kw_only=True)
    # Everything else is set automatically
    source_type: str = field(default="<memory>", init=False)

    def __post_init__(self) -> None:
        if self.source_uri == "UNSET":
            object.__setattr__(self, "source_uri", f"{id(self)}")

        super().__post_init__()

    def get_raw(self) -> str | None:
        return self._raw


@dataclass(frozen=True)
class FileSource(Source):
    """Represents a file source."""

    relative_path: Path
    # Everything else is set automatically
    source_uri: str = field(default="UNSET", init=False)
    source_type: str = field(default="File", init=False)
    _raw: t.Any | None = field(default=None, compare=False, repr=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_uri", self.fqn)
        super().__post_init__()

    def _load_raw(self) -> None:
        if self.relative_path.exists() and self.relative_path.is_file():
            try:
                object.__setattr__(self, "_raw", self.relative_path.read_bytes())
            except Exception:
                logger.exception(f"Failed to read file {self.relative_path}")

    @property
    def file_name(self) -> str:
        return self.relative_path.name

    @property
    def fqn(self) -> str:
        return self.relative_path.as_posix()

    def get_raw(self) -> t.Any | None:
        if self._raw is None:
            self._load_raw()

        return self._raw


@dataclass(frozen=True)
class ZippedFileSource(FileSource):
    """Represents a file source that is archived in a zip file."""

    in_zip_path: Path

    def _load_raw(self) -> None:
        if self.relative_path.exists() and self.relative_path.is_file():
            try:
                with zipfile.ZipFile(self.relative_path) as zip_file:
                    object.__setattr__(self, "_raw", zip_file.read(str(self.in_zip_path)))
            except Exception:
                logger.exception(f"Failed to read zip file {self.relative_path}")

    @property
    def fqn(self) -> str:
        return f"{self.relative_path.as_posix()}{URI_DELIM}{self.in_zip_path.as_posix()}"


@dataclass(frozen=True)
class TextFileSource(FileSource, TextSource):
    """Represents a text file source."""

    def _load_raw(self) -> None:
        if self.relative_path.exists() and self.relative_path.is_file():
            try:
                object.__setattr__(self, "_raw", read_text_unknown_encoding(self.relative_path))
            except Exception:
                logger.exception(f"Failed to read file {self.relative_path}")

    def get_raw(self) -> str | None:
        if self._raw is None:
            self._load_raw()

        return self._raw


# -----------------------------------------------------------------------------
# ----------------------------- Position Classess ---------------------------------


@dataclass(frozen=True)
class EntireSourcePosition(Position):
    """A singelton class representing a position with boundaries indicating whole source was
    used."""

    _instance: t.ClassVar[EntireSourcePosition | None] = None

    def __new__(cls) -> EntireSourcePosition:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    @property
    def fqn(self) -> str:
        return "(entire source)"

    def __str__(self) -> str:
        return self.fqn

    def __repr__(self) -> str:
        return "EntireSourcePosition()"


@dataclass(frozen=True)
class PositionSet(Position):
    """Special positon type that represents a set of positions."""

    positions: tuple[Position, ...]

    @property
    def fqn(self) -> str:
        return f"PositionSet({SETS_DELIM.join([s.fqn for s in self.positions])})"

    def __contains__(self, item: Position) -> bool:
        return item in self.positions

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self) -> t.Iterator[Position]:
        return iter(self.positions)

    def __getitem__(self, key: int) -> Position:
        return self.positions[key]


@dataclass(frozen=True)
class XMLPath(Position):
    """Base class for all XML position types."""

    xpath: str

    @property
    def fqn(self) -> str:
        return self.xpath


@dataclass(frozen=True)
class CodePoint(DataClassSerializeMixin):
    """Code point (absolute index, line, column) in a text source."""

    index: int
    line: int
    column: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be >= 0")

        if self.line < 1:
            raise ValueError(f"Line number must be >= 1, got {self.line}")

        if self.column < 0:
            raise ValueError(f"Column number must be >= 0, got {self.column}")

    def __lt__(self, other: CodePoint) -> bool:
        if not isinstance(other, CodePoint):
            raise NotImplementedError()

        return self.index < other.index

    def __le__(self, other: CodePoint) -> bool:
        if not isinstance(other, CodePoint):
            raise NotImplementedError()

        return self.index <= other.index

    def __str__(self) -> str:
        return f"L{self.line}:C{self.column}"

    def __repr__(self) -> str:
        return f"CodePoint({self.line=}, {self.column=}, {self.index=})"

    def rebase(self, base: CodePoint) -> CodePoint:
        """Recalculate code point position assuming it was relative to a given base starting point
        and return as CodePoint."""
        if self.line == 1:
            column = self.column + base.column
        else:
            column = self.column

        return CodePoint(self.index + base.index, self.line + base.line - 1, column)


@dataclass(frozen=True)
class CodeRange(Position):
    """Represents a code range within a plain text source.

    Attributes:
        start (CodePoint): The start point of the range (the first character included in the range).
        end (CodePoint): The end point of the range (the first character after the code that's part of the range).

    """

    start: CodePoint
    end: CodePoint

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Start point {self.start} must be <= end point {self.end}")

    def overlaps(self, other: CodeRange) -> bool:
        """Returns True if the two ranges overlap, including when ther are adjacent."""
        return self.end >= other.start and self.start <= other.end

    def __contains__(self, other: CodeRange) -> bool:
        if not isinstance(other, CodeRange):
            raise NotImplementedError()

        return self.start <= other.start and other.end <= self.end

    def __lt__(self, other: CodeRange) -> bool:
        if not isinstance(other, CodeRange):
            raise NotImplementedError()

        return self.end < other.start

    def __le__(self, other: CodeRange) -> bool:
        if not isinstance(other, CodeRange):
            raise NotImplementedError()

        return self.end <= other.start

    def __add__(self, other: CodeRange) -> CodeRange:
        if not isinstance(other, CodeRange):
            raise NotImplementedError()

        return CodeRange(start=min(self.start, other.start), end=max(self.end, other.end))

    def __str__(self) -> str:
        return f"L{self.start.line}C{self.start.column}-L{self.end.line}C{self.end.column}"

    @property
    def fqn(self) -> str:
        return f"{self.start.index}-{self.end.index}"


def get_code_range(
    start_index: int,
    start_line: int,
    start_column: int,
    end_index: int,
    end_line: int,
    end_column: int,
) -> CodeRange:
    """Returns a code range from the given start and end points.

    Args:
        start_index (int): start index.
        start_line (int): start line.
        start_column (int): start column.
        end_index (int): end index.
        end_line (int): end line.
        end_column (int): end column.

    Returns:
        CodeRange: code range.

    """

    return CodeRange(
        start=CodePoint(index=start_index, line=start_line, column=start_column),
        end=CodePoint(index=end_index, line=end_line, column=end_column),
    )


EMPTY_CODE_RANGE = get_code_range(0, 1, 0, 0, 1, 0)
# -----------------------------------------------------------------------------
# ----------------------------- Origin Classes ---------------------------------


@dataclass(frozen=True)
class MultiOrigin(Origin):
    """Represents multiple origins."""

    origins: t.Sequence[Origin]
    # Inferrred from origins
    position: PositionSet = field(init=False)
    # If all origins have the same source, this is the source, otherwise it's a SourceSet
    # Ibferred from origins
    source: SourceSet | Source = field(init=False)

    def __post_init__(self) -> None:
        if len(self.origins) < 2:
            raise ValueError("MultiOrigin must have at least two origin")

        if all(origin.source == self.origins[0].source for origin in self.origins[1:]):
            object.__setattr__(self, "source", self.origins[0].source)
        else:
            object.__setattr__(
                self,
                "source",
                SourceSet(tuple([origin.source for origin in self.origins])),
            )

        object.__setattr__(
            self,
            "position",
            PositionSet(tuple([origin.position for origin in self.origins])),
        )

    def get_raw(self) -> list[t.Any | None]:
        return [origin.get_raw() for origin in self.origins]


@dataclass(frozen=True)
class XMLFileOrigin(Origin):
    """Represents an single tag, attribute or tag text value in an XML file.

    get_raw() always returns None.

    """

    source: Source
    position: XMLPath

    def get_raw(self) -> None:
        return None


@dataclass(frozen=True)
class CodeOrigin(Origin):
    source: Source
    position: CodeRange

    def get_raw(self) -> str | None:
        """Returns the code chunk inside this range."""
        code = self.source.get_raw()

        if not isinstance(code, str):
            return None

        return code[self.position.start.index : self.position.end.index]

    def __add__(self, other: Origin) -> Origin:
        if not isinstance(other, CodeOrigin):
            return super().__add__(other)

        if self.source != other.source or not self.position.overlaps(other.position):
            return super().__add__(other)

        return CodeOrigin(source=self.source, position=self.position + other.position)


@dataclass(frozen=True)
class GeneratedCodeOrigin(CodeOrigin):
    """Represents a code origin that was generated by the parser, thus have a source.

    Position is always EMPTY_CODE_RANGE (0-0 index). get_raw() always returns None.

    """

    source: Source
    position: CodeRange = field(default=EMPTY_CODE_RANGE, init=False)

    def get_raw(self) -> None:
        return None


# -----------------------------------------------------------------------------
# ----------------------------- Helpers ---------------------------------


def get_xml_origin(source_file_or_src: Path | Source, xpath: str) -> XMLFileOrigin:
    """Returns an XML origin from the given FileSource or source file path (creating FileSource on
    the fly) and xpath."""
    if isinstance(source_file_or_src, Path):
        source: Source = FileSource(source_file_or_src)
    else:
        source = source_file_or_src

    return XMLFileOrigin(
        source=source,
        position=XMLPath(xpath),
    )


def merge_origins(*origins: Origin) -> Origin:
    """Merges a list of origins into a single multi origin."""

    if any(not isinstance(origin, Origin) for origin in origins):
        raise ValueError("Cannot merge non-origin objects")

    if len(origins) == 1:
        return origins[0]

    new_origins: list[Origin] = []

    for origin in origins:
        if isinstance(origin, NoOrigin):
            continue
        elif isinstance(origin, MultiOrigin):
            new_origins.extend(origin.origins)
        else:
            new_origins.append(origin)

    if len(new_origins) == 0:
        return NoOrigin()

    if len(new_origins) == 1:
        return new_origins[0]

    return MultiOrigin(origins=new_origins)


def concat_origins(origin: Origin, *origins: Origin) -> Origin:
    """Similar to merge_origins but uses addition which in case of CodeOrigin may merge overlapping
    ranges into a single CodeOrigin.

    Returns:
        Origin: new origin.

    """
    if len(origins) == 0:
        return origin

    new_origin = origin
    for origin in origins:
        new_origin += origin

    return new_origin
