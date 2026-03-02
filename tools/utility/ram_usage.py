import time
import psutil

def _get_available_ram_gb() -> float:
   """Return available RAM in GB."""
   return psutil.virtual_memory().available / (1024 ** 3)

def wait_for_ram(threshold_gb: float, check_interval: float) -> None:
   """Block until at least threshold_gb of RAM is available."""
   while _get_available_ram_gb() < threshold_gb:
      print(f'Waiting for RAM... Available: {_get_available_ram_gb():.2f} GB, Required: {threshold_gb:.2f} GB')
      time.sleep(check_interval)