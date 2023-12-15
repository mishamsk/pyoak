# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Added

- TBD

### ✨ Changed

- TBD

### ⚠️ Breaking

- TBD

### 🐛 Fixed

- TBD

### 🔥 Removed

- TBD

### 📖 Documentation

- TBD

### 🛠️ Internal

- TBD

## [1.2.0]

This mainly a backport of enhanced XPath & Pattern Matcher from 2.0.0, but also makes serialization dependencies options as well as removes the ability of TextFileSources to guess file encoding (and chardet dependency with it).

This release was entirely contributed by @dmmoeu. Thanks a lot!

### 🚀 Added

- ASTNode `find` and `findall` API's (backport from v2). See [README](README.md#xpath) for details.

### ✨ Changed

- pyyaml, orjson, msgpack are now optional dependencies. They can be installed via a new set of extras: msgpack, orjson, yaml or all. E.g.: `pip install pyoak[msgpack]`. `DataClassSerializeMixin` and hence `ASTNode` will have `to/from_yaml` and `to/from_msgpack` methods if the corresponding dependencies are installed. `to/from_json` methods are always available and use `json` if `orjson` is not installed.

### ⚠️ Breaking

- Pattern Grammar is now the same as v2 series. This is a breaking change, check out the [README](README.md#pattern-matching) for details.
- `TextFileSource` no longer guesses file encoding. This is a breaking change, but it's a good thing. If you need to read a file with a specific encoding, read it yourself and pass the content as a string to the `TextFileSource` constructor's `_raw` attribute.

### 🐛 Fixed

- A couple of bugs in v2 pattern matcher were fixed in this release.

## [1.1.1] - 2023-08-02

Quick hotfix release to restore backwards compatibility with 1.0.0

### 🐛 Fixed

- 1.1.0 wouldn't deserialize NoSource/NoPosition/NoOrigin serialized by an earlier version of the library. This is now fixed and the lib should be fully backwards compatible.

## [1.1.0] - 2023-08-02

### 🚀 Added

- Instantiated singleton constants NO_ORIGIN, NO_SOURCE, NO_POSITION in `pyoak.origin`

### ✨ Changed

- Removes unnecessary need to pass deserialization option to distinguish whether Sources were serialized as normal or as registry id's
- Serializes NoXXX classes as empty dicts now

### ⚠️ Breaking

- NoSource singleton instance is not part of the Source registry anymore and reports source_registry_id as -1. However it is still returned via `Source.list_registered_sources()` API for backwards compatibility.

### 🛠️ Internal

- ⬇️ relaxed library dependencies

## [1.0.0] - 2023-07-29

### Added

- First public release.
