from model.model import ThermalConductivityModel

# Wire the concrete model for the runner. This is the single tool-owned wiring
# point — the runner imports `handler` from here.
handler = ThermalConductivityModel()
