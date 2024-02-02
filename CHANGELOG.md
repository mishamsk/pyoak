# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Added

- Gracefully handle string annotation resolution errors in visitor
- ASTXpath now support equality comparison
- Allow tree alternatives in xpath

### ✨ Changed

- Use regular dict for visitor dispatch cache. Weakref is not needed, since methods will always be alive as long as the class is alive.
- Clarify visitor dispatch cache behavior & limitations in docstring
- Expand visitor tests

### ⚠️ Breaking

- TBD

### 🐛 Fixed

- ASTXpath cache accounts for types mapping that was passed to it
- Xpathes will now forbid incorrect order and duplicate items in element definitions
- Fix ASTVisitor dispatching for multiple inheritance

### 🔥 Removed

- TBD

### 📖 Documentation

- TBD

### 🛠️ Internal

- TBD

## [3.0.0a10]

Highlight: visitor base class improvements

### 🚀 Added

- add py312 to tox tests & workflows
- make a standalone from_pattern
- tests for caching and from_pattern

### ✨ Changed

- use custom cache to preserve from_pattern signature for IDEs

### 🛠️ Internal

- in traversal bench use python timeit default formatting

## [3.0.0a9]

Highlight: visitor base class improvements

### 🚀 Added

- ⏱️ added visitor method benchmark

### ✨ Changed

- BREAKING: visitor now matches nodes to methods by type annotation, ignoring method name
- visitor method matching is factored out allowing to create subclasses with visit method signatures having additional arguments
- ⏱️ in matcher bench - do right most leaf search and compare to traversal

## [3.0.0a8]

Highlight: pattern aliases with recursion and regex like wildcards!

### 🚀 Added

- Introduced pattern aliases with recursion
- Added pattern wildcards (qualifiers)
- Allowed empty class and thus simpler `()` (instead of `(*)`) for any node. Although the older syntax is still supported.
- Added 'with', 'as' tokens and tests to pattern lexer (for with patterns)
- Added optional boundaries for pattern alternatives
- Allowed capturing at the outmost level in patterns
- Added ability to capture multiple values under same name in a pattern

### ✨ Changed

- BREAKING: Disallowed non-existent field names in node patterns - raises an error now
- BREAKING: Commas now enforced in sequence patterns
- Show meaningful error if pattern definition is not a string
- Showed meaningful error if string value pattern is not a valid regex
- Improved error reporting for None value in pattern
- Provided better context in lexer errors

### 🐛 Fixed

- Do not allow values after any wildcard in sequence patterns
- Made sure the outer captured value stored, not inner

### 📖 Documentation

- Added full pattern syntax to README

### 🛠️ Internal

- Simplified sequence/value grammar
- Unified capture key parsing logic
- Updated match bench with recursive pattern test
- Extended and improved pattern test structure
- Made match dict tests more robust, compare entire dict
- Multiple improvements to parser error messages

## [3.0.0a7]

### 🚀 Added

- add pointer to precise location in parser errors

## [3.0.0a6]

### 🚀 Added

- add optional ancestors kwarg to xpath.match to use precomputed ancestors

### ✨ Changed

- allow pattern as the right most element in xpath

### 🛠️ Internal

- update pre-commit hooks

- explain lexer mode boilerplate better

## [3.0.0a5]

### ✨ Changed

- allow capture key to be re-used (last captured value wins)
- improve CNAME token error message

### 🛠️ Internal

- explain lexer mode boilerplate better

## [3.0.0a4]

### 🚀 Added

- source registry tests
- expose Source.get_source_by_registry_id API

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
