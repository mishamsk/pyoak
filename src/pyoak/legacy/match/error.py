class ASTPatternDefinitionError(Exception):
    """Base class for all tree pattern matching rule definition errors."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class ASTPatternDidNotMatchError(Exception):
    """Used to stop matching when class mismatched."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class ASTXpathDefinitionError(Exception):
    """Base class for all xpath rule definition errors."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message
