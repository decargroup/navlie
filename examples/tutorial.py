import navlie as nav
from navlie.lib import VectorState

x0 = nav.lib.VectorState([0, 0, 0], stamp=0.0)


class BicycleModel(nav.ProcessModel):
    def evaluate(
        self, x: VectorState, u: nav.VectorInput, dt: float
    ) -> VectorState:
        pass
