from typing import Protocol, runtime_checkable

@runtime_checkable
class PrintableMetric(Protocol):
    def print_results(self) -> None:
        pass
    def to_dict(self) -> dict:
        pass