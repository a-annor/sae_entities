"""Error classes for config2py."""


class Config2PyError(Exception):
    """Base class for config2py errors."""


class ConfigNotFound(Config2PyError):
    """Raised when a config file is not found."""
