from __future__ import annotations

import enum
import inspect
from dataclasses import Field
from datetime import date, datetime, time
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Type, TypeVar, cast
from uuid import UUID

from mashumaro.config import ADD_DIALECT_SUPPORT, BaseConfig
from mashumaro.dialect import Dialect
from mashumaro.exceptions import InvalidFieldValue
from mashumaro.helper import pass_through
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.types import SerializableType

_HAS_MSGPACK = False
try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    pass

json_dialect: Type[Dialect] | None = None
try:
    import orjson

    def json_loader(value: bytes | str) -> dict[str, Any]:
        return cast(dict[str, Any], orjson.loads(value))

    def json_dumper_to_byte(value: dict[str, Any], indent: bool = False) -> bytes:
        if indent:
            return orjson.dumps(
                value,
                option=orjson.OPT_INDENT_2,
            )
        else:
            return orjson.dumps(value)

    def json_dumper_to_str(value: dict[str, Any], indent: bool = False) -> str:
        if indent:
            return orjson.dumps(
                value,
                option=orjson.OPT_INDENT_2,
            ).decode(encoding="utf-8")
        else:
            return orjson.dumps(value).decode(encoding="utf-8")

    class OrjsonDialect(Dialect):
        serialization_strategy = {  # noqa: RUF012
            datetime: {"serialize": pass_through},
            date: {"serialize": pass_through},
            time: {"serialize": pass_through},
            UUID: {"serialize": pass_through},
        }

    json_dialect = OrjsonDialect
except ImportError:
    import json

    def json_loader(value: bytes | str) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(value))

    def json_dumper_to_byte(value: dict[str, Any], indent: bool = False) -> bytes:
        if indent:
            return json.dumps(value, indent=2).encode(encoding="utf-8")
        else:
            return json.dumps(value).encode(encoding="utf-8")

    def json_dumper_to_str(value: dict[str, Any], indent: bool = False) -> str:
        if indent:
            return json.dumps(value, indent=2)
        else:
            return json.dumps(value)


_HAS_YAML = False
try:
    import ruamel.yaml
    from ruamel.yaml.compat import StringIO

    ruamel_yaml = ruamel.yaml.YAML(typ="rt", pure=False)

    def yaml_dumper(value: dict[str, Any]) -> str:
        stream = StringIO()
        ruamel_yaml.dump(value, stream)
        return stream.getvalue()  # type: ignore

    def yaml_loader(value: bytes | str) -> dict[str, Any]:
        return ruamel_yaml.load(value)  # type: ignore

    _HAS_YAML = True
except ImportError:
    try:
        import yaml

        YamlLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        YamlDumper = getattr(yaml, "CDumper", yaml.Dumper)

        def yaml_dumper(value: dict[str, Any]) -> str:
            return yaml.dump(value, Dumper=YamlDumper)

        def yaml_loader(value: bytes | str) -> dict[str, Any]:
            return yaml.load(value, Loader=YamlLoader)  # type: ignore

        _HAS_YAML = True
    except ImportError:
        pass


TYPES: dict[str, Type[DataClassSerializeMixin]] = {}

T = TypeVar("T", bound="DataClassSerializeMixin")


class MessagePackDialect(Dialect):
    serialization_strategy = {  # noqa: RUF012
        bytes: pass_through,
        bytearray: {
            "deserialize": bytearray,
            "serialize": pass_through,
        },
    }


@enum.unique
class SerializationOption(str, enum.Enum):
    SKIP_CLASS = "skip_class"
    SORT_KEYS = "sort_keys"


TYPE_KEY = "__type"
"""Key used to store/read DataClassSerializeMixin type during (de)serialization."""


class DataClassSerializeMixin(DataClassDictMixin, SerializableType):
    __slots__ = ()

    __serialization_options: ClassVar[dict[str, Any]] = {}
    __mashumaro_dialect: ClassVar[Type[Dialect] | None] = None

    class Config(BaseConfig):
        serialization_strategy = {  # noqa: RUF012
            Path: {"serialize": lambda x: x.as_posix()},
            Field[Any]: {"serialize": lambda _: None, "deserialize": lambda _: None},
        }
        code_generation_options = [ADD_DIALECT_SUPPORT]  # type: ignore   # noqa: RUF012

    def _get_serialization_options(self) -> dict[str, Any]:
        return DataClassSerializeMixin.__serialization_options

    @classmethod
    def _get_deserialization_options(cls) -> dict[str, Any]:
        return DataClassSerializeMixin.__serialization_options

    def _get_serialization_mashumaro_dialect(self) -> Type[Dialect] | None:
        return DataClassSerializeMixin.__mashumaro_dialect

    @classmethod
    def _get_deserialization_mashumaro_dialect(cls) -> Type[Dialect] | None:
        return DataClassSerializeMixin.__mashumaro_dialect

    def __post_serialize__(self, d: dict[str, Any]) -> dict[str, Any]:
        skip_class = self._get_serialization_options().get(SerializationOption.SKIP_CLASS, False)
        sort_keys = self._get_serialization_options().get(SerializationOption.SORT_KEYS, False)

        out = {}

        if not skip_class:
            # Add class name
            out[TYPE_KEY] = self.__class__.__name__

        if sort_keys:
            # Output keys in sorted order for stable serialization
            for k, v in sorted(d.items(), key=itemgetter(0)):
                out[k] = v
        else:
            out.update(d)

        return out

    def _serialize(self) -> dict[str, Any]:
        if DataClassSerializeMixin.__mashumaro_dialect is not None:
            return self.to_dict(dialect=DataClassSerializeMixin.__mashumaro_dialect)
        else:
            return self.to_dict()

    @classmethod
    def _deserialize(cls: Type[T], value: dict[str, Any]) -> T:
        class_name = value.get(TYPE_KEY)

        if isinstance(class_name, str):
            clazz = TYPES.get(class_name, None)
        else:
            clazz = cls

        if clazz is None:
            raise ValueError(f"Unknown class name: {class_name}")

        if DataClassSerializeMixin.__mashumaro_dialect is not None:
            return cast(
                T,
                clazz.from_dict(value, dialect=DataClassSerializeMixin.__mashumaro_dialect),
            )
        else:
            return cast(T, clazz.from_dict(value))

    def __init_subclass__(cls, **kwargs: Any):
        if TYPES.get(cls.__name__) is not None:
            # We are not allowed to redefine a class that is already defined in a different package
            # except for one case - when dataclass re-creates the class with __slots__.
            new_module = inspect.getmodule(cls)
            old_module = inspect.getmodule(TYPES[cls.__name__])

            if new_module is not old_module:
                raise ValueError(
                    f"DataClassSerializeMixin subclass <{cls.__name__}> is already defined "
                    f"in {old_module!s}. Please use a different name."
                )

        TYPES[cls.__name__] = cls
        return super().__init_subclass__(**kwargs)

    def as_dict(
        self,
        mashumaro_dialect: Type[Dialect] | None = None,
        serialization_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize this object to a dictionary.

        Args:
            mashumaro_dialect (Dialect, optional): The Mashumaro dialect to use for serialization.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            dict[str, Any]: The serialized object.

        """

        # Store the options and mashumaro dialect for use by subclasses
        if serialization_options is not None:
            DataClassSerializeMixin.__serialization_options.update(serialization_options)
        DataClassSerializeMixin.__mashumaro_dialect = mashumaro_dialect

        try:
            ret = self._serialize()
        finally:
            # Clear the kwargs and dialect
            DataClassSerializeMixin.__serialization_options = {}
            DataClassSerializeMixin.__mashumaro_dialect = None

        return ret

    @classmethod
    def as_obj(
        cls: Type[T],
        value: dict[str, Any],
        *,
        mashumaro_dialect: Type[Dialect] | None = None,
        serialization_options: dict[str, Any] | None = None,
    ) -> T:
        """Deserialize a dictionary to an object.

        Args:
            value (dict[str, Any]): The serialized object.
            mashumaro_dialect (Dialect, optional): The Mashumaro dialect to use for serialization.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            T: The deserialized object.

        """

        # Store the options and mashumaro dialect for use by subclasses
        if serialization_options is not None:
            DataClassSerializeMixin.__serialization_options.update(serialization_options)
        DataClassSerializeMixin.__mashumaro_dialect = mashumaro_dialect

        try:
            ret = cls._deserialize(value)
        finally:
            # Clear the kwargs and dialect
            DataClassSerializeMixin.__serialization_options = {}
            DataClassSerializeMixin.__mashumaro_dialect = None

        return ret

    def to_jsonb(
        self,
        *,
        indent: bool = False,
        serialization_options: dict[str, Any] | None = None,
    ) -> bytes:
        """Serialize this object to JSON (as bytes).

        Mashumaro dialects are not supported for this method.

        Args:
            indent (bool, optional): If True, the JSON will be indented. Defaults to False.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            bytes: The serialized object.

        """
        return json_dumper_to_byte(
            self.as_dict(
                mashumaro_dialect=json_dialect,
                serialization_options=serialization_options,
            ),
            indent=indent,
        )

    def to_json(
        self,
        *,
        indent: bool = False,
        serialization_options: dict[str, Any] | None = None,
    ) -> str:
        """Serialize this object to JSON (as a string).

        Mashumaro dialects are not supported for this method.

        Args:
            indent (bool, optional): If True, the JSON will be indented. Defaults to False.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            str: The serialized object.

        """
        return json_dumper_to_str(
            self.as_dict(
                mashumaro_dialect=json_dialect,
                serialization_options=serialization_options,
            ),
            indent=indent,
        )

    @classmethod
    def from_json(
        cls: Type[T],
        value: bytes | str,
        serialization_options: dict[str, Any] | None = None,
    ) -> T:
        """Deserialize this object from JSON (as bytes or str).

        Mashumaro dialects are not supported for this method.

        Args:
            value (bytes | str): The serialized object.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            T: The deserialized object.

        """

        return cls.as_obj(
            json_loader(value),
            mashumaro_dialect=json_dialect,
            serialization_options=serialization_options,
        )

    if _HAS_MSGPACK:

        def to_msgpck(self, serialization_options: dict[str, Any] | None = None) -> bytes:
            """Serialize this object to MessagePack (as bytes).

            Mashumaro dialects are not supported for this method.

            Args:
                serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
            for customization.

            Returns:
                bytes: The serialized object.

            """
            return cast(
                bytes,
                msgpack.packb(
                    self.as_dict(
                        mashumaro_dialect=MessagePackDialect,
                        serialization_options=serialization_options,
                    ),
                    use_bin_type=True,
                ),
            )

        @classmethod
        def from_msgpck(
            cls: Type[T], value: bytes, serialization_options: dict[str, Any] | None = None
        ) -> T:
            """Deserialize this object from MessagePack (as bytes).

            Mashumaro dialects are not supported for this method.

            Args:
                value (bytes): The serialized object.
                serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
            for customization.

            Returns:
                T: The deserialized object.

            """

            return cls.as_obj(
                msgpack.unpackb(value, raw=False),
                mashumaro_dialect=MessagePackDialect,
                serialization_options=serialization_options,
            )

    if _HAS_YAML:

        def to_yaml(
            self,
            mashumaro_dialect: Type[Dialect] | None = None,
            serialization_options: dict[str, Any] | None = None,
        ) -> str:
            """Serialize this object to YAML (as a string).

            Args:
                mashumaro_dialect (Dialect, optional): The Mashumaro dialect to use for serialization.
                serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
            for customization.

            Returns:
                str: The serialized object.

            """
            return yaml_dumper(
                self.as_dict(
                    mashumaro_dialect=mashumaro_dialect,
                    serialization_options=serialization_options,
                )
            )

        @classmethod
        def from_yaml(
            cls: Type[T],
            value: str | bytes,
            mashumaro_dialect: Type[Dialect] | None = None,
            serialization_options: dict[str, Any] | None = None,
        ) -> T:
            """Deserialize this object from YAML (as a string).

            Args:
                value (str): The serialized object.
                mashumaro_dialect (Dialect, optional): The Mashumaro dialect to use for serialization.
                serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
            for customization.

            Returns:
                T: The deserialized object.

            """

            return cls.as_obj(
                yaml_loader(value) or {},
                mashumaro_dialect=mashumaro_dialect,
                serialization_options=serialization_options,
            )


def unwrap_invalid_field_exception(exc: InvalidFieldValue) -> tuple[str, BaseException]:
    """Takes InvalidFieldValue exception and returns a tuple of a path to the nested field that
    caused the first non InvalidFieldValue exception as well as the actual exception object.

    If the exception doesn't contain any nested exceptions, the path will be the same as the field
    name and the exception object will be the same as the one passed in.

    """
    path = exc.field_name
    out_exc: BaseException = exc

    while out_exc.__context__ is not None:
        out_exc = out_exc.__context__
        if isinstance(out_exc, InvalidFieldValue):
            path += f".{out_exc.field_name}"
        else:
            break

    return path, out_exc
