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

## [3.0.0a3]

Highlight: an alternative (OR) pattern for trees.

### 🚀 Added

- Introduced an alternative (OR) pattern for trees
- More documentation on pattern matcher

### ✨ Changed

- Moved `from_pattern` factory to BaseMatcher

### 🔥 Removed

- Breaking: Removed MultiPatternMatcher in favor of OR pattern

### 🛠️ Internal

- Improved pattern matching with the use of `lru_cache` instead of custom cache
- Configured mypy to not follow imports for traitlets

## [3.0.0a2]

### ✨ Changed

- Improve parser error formatting

## [3.0.0a1]

This is the first early alpha of the next version that combines the best of v1 and v2. We are expanding core features, focusing on performance for huge trees, and eliminating everything that can be easily added externally.

This version already includes significant changes and improvements, including the removal of certain dependencies (most notably - pattern parser is now bespoke and lark is not a dependency anymore), backporting of method code generation from v2, and several breaking changes.

### 🚀 Added

- Unified grammar errors under single ASTXpathOrPatternDefinitionError
- Added ability to pass types to check against when building Xpath and patterns
- Backported remaining node tests
- Backported codegen and support for slots
- Added optional comma to sequence pattern

### ✨ Changed

- Rewrote xpath/pattern to bespoke parser
- breaking: backport strict typing to ast node
- backport opt-in runtime type checks
- Moved helpers, visitors to separate modules
- NodeMatcher.from_pattern now raises on error, similar to ASTXPath
- Backported repr_fn for ASTNode and separate config module
- Backported get_property_fields API (plain Field iterable)
- Backported sort_keys in various field getter APIs
- Allowed slotted DataClassSerializeMixin subclasses

### 🐛 Fixed

- TextFileSource no longer tries reading from non-existent file more than once

### 🔥 Removed

- Removed lark as a dependency
- Removed ASTXpathDefinitionError and ASTPatternDefinitionError
- Removed old, string based xpath & calculate_xpath API
- Removed ASTTransormer

### 📖 Documentation

- Updated readme

### 🛠️ Internal

- Allow pre-releases in bump2version
- Backported and improved benchmarks from v2
- Backported typing modules from v2
- Updated minimum development dependencies
- Stricter mypy checks and ensured ruff targets py3.10
- Switched to taskdev from makefile
- Added sentinel for unset source_uris
- Added subclassing violation test and slotted subclass test
- Added TextFileSource failed file read test

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
