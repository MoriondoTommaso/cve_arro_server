from __future__ import annotations

from fastapi import HTTPException, status


class DatasetNotFound(HTTPException):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, f"Dataset not found: {dataset_id}")


class InvalidSlice(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status.HTTP_400_BAD_REQUEST, f"Invalid slice: {detail}")


class DatasetNotSliceable(HTTPException):
    """Raised when a dataset exists but cannot be sliced (e.g. 0-d or group)."""

    def __init__(self, dataset_id: str, reason: str) -> None:
        super().__init__(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Dataset '{dataset_id}' cannot be sliced: {reason}",
        )


class OptionalDependencyMissing(HTTPException):
    def __init__(self, package: str, feature: str) -> None:
        super().__init__(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Optional dependency '{package}' required for {feature} is not installed.",
        )


class MetadataUnavailable(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, f"Metadata unavailable: {detail}")
