"""Exception types with user-friendly messages. Technical details go to the log."""
from __future__ import annotations


class GeoDubberError(Exception):
    """Base class. `user_message` is shown in the GUI, str(exc) goes to the log."""

    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class MissingDependencyError(GeoDubberError):
    pass


class UnsupportedFormatError(GeoDubberError):
    pass


class CorruptMediaError(GeoDubberError):
    pass


class MediaProcessingError(GeoDubberError):
    pass


class ModelMissingError(GeoDubberError):
    pass


class VoiceUnavailableError(GeoDubberError):
    pass


class InsufficientResourcesError(GeoDubberError):
    pass


class DownloadError(GeoDubberError):
    pass


class CancelledError(GeoDubberError):
    def __init__(self, message: str = "Cancelled by user"):
        super().__init__(message, message)
