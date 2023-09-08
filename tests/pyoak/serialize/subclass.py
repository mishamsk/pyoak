from dataclasses import dataclass

from pyoak.serialize import DataClassSerializeMixin


@dataclass(frozen=True)
class SerializeTest(DataClassSerializeMixin):
    attr: str
