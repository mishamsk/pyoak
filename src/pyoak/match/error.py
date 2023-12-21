class ASTPatternDidNotMatchError(Exception):
    """Used to stop matching when class mismatched."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class ASTXpathOrPatternDefinitionError(Exception):
    """Base class for all xpath or tree patterns rule definition errors."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message
