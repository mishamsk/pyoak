# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ✨ Highlights
- TBD

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

## [2.1.0] - 2023-09-11

A quick iteration on the 2.0.0 release to ensure consistency of the API's naming and future support for slotted classes.

### ⚠️ Breaking

- Unify Tree API's to start with `get_` or `is_`

### 🐛 Fixed

- Workarounds for a couple stdlib slotted dataclasses bugs to support slotted ASTNodes (still contingent on Mashumaro update)

### 🛠️ Internal

- ⬇️ relaxed library dependency for Mashumaro
- ⬆️ dev deps
- Switch black pre-commit hook to new mirror repo

## [2.0.0] - 2023-09-03

This is a significant rework, that brings most of the features previously on the roadmap, but also introduces major breaking changes. To avoid disruption, the v1 code has been moved to `legacy` sub-package and the legacy `ASTNode` is now called `AwareASTNode`. Legacy package will emit deprecation warnings on import though.

### ✨ Highlights
* Ability to re-use nodes across multiple ASTs without creating duplicates
* Nodes are now hashable!
* 100x faster pattern matching with support for pattern variables & simplified grammar
* ~50% faster traversal
* XPath search APIs (not unlike those in ElementTree)
* Opt-in runtime type checking of AST nodes
* ... and more

### 🚀 Added

- `pyoak.tree.Tree` class that provides API for to retrieve node parents in a tree as well as traversing "up" the tree
- `pyoak.config` module that provides global configuration options
- Opt-in runtime type checking at node creation via `pyoak.config.RUNTIME_TYPE_CHECK` flag
- XPath
    - `ASTXpath.findall` and sister `ASTNode.findall`, `ASTNode.find` methods. Work similarly to ElementTree's `findall` and `find` methods. The latter accept both a compiled `ASTXpath` instnace or string.
- Pattern Matching
    - Ability to match against multiple types: `(Type1|Type2|... ...)`
    - Pattern variables: `(* @field_val_from -> val @field_compare_to = $val)`

### ✨ Changed

- Traversal methods `dfs`, `bfs` and `gather` do not return the node itself anymore
- `ASTVisitor` and `ASTTransformVisitor` are now in a separate module `pyoak.visitor`
- Auto-generated ID hash size is now configurable via `config.ID_DIGEST_SIZE`, and defaults to 8 bytes (16 hex chars)
- Optional trace logging switch is now in `config` module
- `ASTNode._iter_child_fields` propoted to a public API `ASTNode.iter_child_fields`

### ⚠️ Breaking

- Traversal methods `dfs`, `bfs` and `gather` return a tuple of `(node, parent, field, index)` instead of just the node
- All methods of `ASTNode` that worked by going "up" the tree are now in `Tree` class
- Filters & pruners for traversal methods now must accept a tuple of `(node, parent, field, index)` instead of just the node
- Mixed fields (i.e. with types like `str | ASTNode`) are not allowed anymore. Separate properties and children into separate fields instead.
- Pattern Match grammar has been simplified:
    - any field rule is denoted just by `@field_name=...` instead of `@[child_field=...] #[property=...]` syntax.
    - there are no more `!` and `!!` flags, since sequence matching is now always "strict" (i.e. `@field_name=[]` matches only if the field is a sequence and elements match the pattern exactly)

### 🐛 Fixed

- CI/CD pipeline now correctly runs tests on all platforms
- Local docformatter now uses the same parameters as the one in CI/CD pipeline

### 🔥 Removed

- Runtime checks during traversal/retrieving children. All the decisions whether a field is a child field and/or is a sequence are made based type annotations at class creation
- Removed the ability to provide custom `id` to `ASTNode` constructor. If you need to track source/natural ID's, use a dedicated field for that.
- `id_collision_with` and `original_id` auto-properties. Now if you need to track collisions, this must be done manually.
- `parent`, `parent_field` and `parent_index` auto-properties. Nodes are not aware of the parent anymore. Use `Tree` API instead.
- `ensure_unique_id` and `create_detached` removed as they are no longer needed
- `replace` method. Use the type-checked (as of mypy 1.5+) `dataclasses.replace` instead.
- `replace_with` method, since node's are not parent aware anymore. Use `ASTTransformVisitor` instead.
- `xpath` property and `calcualte_xpath` method. Use `Tree` instead.
- `detached`, `is_attached_root` and `is_attached_subtree` properties. No longer relevant.

### 📖 Documentation

- Updated README

### 🛠️ Internal

- Prepare to support slotted classes (contingent on Mashumaro update)

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
