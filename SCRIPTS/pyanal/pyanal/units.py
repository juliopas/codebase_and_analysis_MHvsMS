from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnitSystem:
    scales: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    name:   str  = ""

    def scale(self, dim: str) -> float:
        return self.scales.get(dim, 1.0)

    def label(self, dim: str) -> str:
        return self.labels.get(dim, "")


SIM = UnitSystem(name="sim")
