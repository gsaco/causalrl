# Deprecation Policy

CausalRL follows a predictable deprecation policy, even pre-1.0, to keep the
public API stable and auditable.

## Timeline

1. **Deprecation warning**: Deprecated features emit warnings for at least one
   minor release before removal.
2. **Removal**: The feature is removed only after the warning window.

## What counts as public

Only the items listed in the Public API page are guaranteed to follow this
policy. Anything else may change without notice.

## How we signal changes

- Runtime warnings using `DeprecationWarning` (or `FutureWarning` when needed).
- Changelog entries describing the change and migration guidance.
