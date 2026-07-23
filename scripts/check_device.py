"""Print the accelerator chosen by the optional training stack."""

from stag_hunt_lang.device import resolve_torch_device

print(resolve_torch_device())

