"""Shared error hierarchy. All lyricsmith exceptions inherit from
LyricsmithError so the CLI can catch them at one boundary (see
ARCHITECTURE.md section 8, failure isolation)."""


class LyricsmithError(Exception):
    """Base class for all expected lyricsmith failures."""


class ConstraintError(LyricsmithError):
    """Raised when a constraint scaffold cannot be built (e.g. unknown genre)."""


class GenerationError(LyricsmithError):
    """Raised when a LineGenerator fails in a way that can't be retried."""


class ValidationError(LyricsmithError):
    """Raised when validating a line/song against constraints fails hard
    (most validation failures are non-fatal ValidationResults, not
    exceptions -- this is reserved for programmer-error-shaped failures)."""
