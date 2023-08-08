# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Instantiated singleton constants NO_ORIGIN, NO_SOURCE, NO_POSITION in `pyoak.origin`

### Changed

- Removes unnecessary need to pass deserialization option to distinguish whether Sources were serialized as normal or as registry id's
- Serializes NoXXX classes as empty dicts now

### Breaking

- NoSource singleton instance is not part of the Source registry anymore and reports source_registry_id as -1. However it is still returned via `Source.list_registered_sources()` API for backwards compatibility.

## [1.0.0] - 2023-07-29

### Added

- First public release.
