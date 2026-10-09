"""Run the pinned upstream converter with a bounded-memory scale-refit adapter."""

import runpy

from exllamav3.modules.quant.exl3_lib import quantize
from refit_scales import refit_scales

quantize.refit_scales = refit_scales
print("Using chunked EXL3 scale refit (no full-head elementwise temporary)", flush=True)
runpy.run_path("/opt/exllamav3/convert.py", run_name="__main__")
