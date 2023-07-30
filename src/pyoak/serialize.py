from __future__ import annotations

import enum
import inspect
from dataclasses import Field
from datetime import date, datetime, time
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Type, TypeVar, cast
from uuid import UUID

import msgpack
import orjson
import yaml
from mashumaro.config import ADD_DIALECT_SUPPORT, BaseConfig
from mashumaro.dialect import Dialect
from mashumaro.exceptions import InvalidFieldValue
from mashumaro.helper import pass_through
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.types import SerializableType

TYPES: dict[str, Type[DataClassSerializeMixin]] = {}

T = TypeVar("T", bound="DataClassSerializeMixin")


class OrjsonDialect(Dialect):
    serialization_strategy = {  # noqa: RUF012
        datetime: {"serialize": pass_through},
        date: {"serialize": pass_through},
        time: {"serialize": pass_through},
        UUID: {"serialize": pass_through},
    }


class MessagePackDialect(Dialect):
    serialization_strategy = {  # noqa: RUF012
        bytes: pass_through,
        bytearray: {
            "deserialize": bytearray,
            "serialize": pass_through,
        },
    }


YamlLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
YamlDumper = getattr(yaml, "CDumper", yaml.Dumper)


@enum.unique
class SerializationOption(str, enum.Enum):
    SKIP_CLASS = "skip_class"
    SORT_KEYS = "sort_keys"


TYPE_KEY = "__type"
"""Key used to store/read DataClassSerializeMixin type during (de)serialization."""


class DataClassSerializeMixin(DataClassDictMixin, SerializableType):
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
        if cls.__name__ in TYPES:
            package = "Unknown"
            module = inspect.getmodule(TYPES[cls.__name__])
            if module is not None:
                package = str(module.__package__)

            raise ValueError(
                f"DataClassSerializeMixin subclass <{cls.__name__}> is already defined in package <{package}>. Please use a different name."
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
        if indent:
            return orjson.dumps(
                self.as_dict(
                    mashumaro_dialect=OrjsonDialect,
                    serialization_options=serialization_options,
                ),
                option=orjson.OPT_INDENT_2,
            )
        else:
            return orjson.dumps(
                self.as_dict(
                    mashumaro_dialect=OrjsonDialect,
                    serialization_options=serialization_options,
                )
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
        return self.to_jsonb(indent=indent, serialization_options=serialization_options).decode(
            encoding="utf-8"
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
            orjson.loads(value),
            mashumaro_dialect=OrjsonDialect,
            serialization_options=serialization_options,
        )

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

    def to_yaml(
        self,
        mashumaro_dialect: Type[Dialect] | None = None,
        serialization_options: dict[str, Any] | None = None,
    ) -> str:
        """Serialize this object to YAML (as a string).

        Mashumaro dialects are not supported for this method.

        Args:
            mashumaro_dialect (Dialect, optional): The Mashumaro dialect to use for serialization.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            str: The serialized object.

        """
        return yaml.dump(
            (
                self.as_dict(
                    mashumaro_dialect=mashumaro_dialect,
                    serialization_options=serialization_options,
                )
            ),
            Dumper=YamlDumper,
        )

    @classmethod
    def from_yaml(
        cls: Type[T],
        value: str | bytes,
        mashumaro_dialect: Type[Dialect] | None = None,
        serialization_options: dict[str, Any] | None = None,
    ) -> T:
        """Deserialize this object from YAML (as a string).

        Mashumaro dialects are not supported for this method.

        Args:
            value (str): The serialized object.
            mashumaro_dialect (Dialect, optional): The Mashumaro dialect to use for serialization.
            serialization_options (dict[str, Any], optional): Additional options that may be accessed by subclasses
        for customization.

        Returns:
            T: The deserialized object.

        """

        return cls.as_obj(
            yaml.load(value, Loader=YamlLoader) or {},
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
