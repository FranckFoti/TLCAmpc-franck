"""Unit tests for BoFLibraryAdapter and BoFRestAdapter.

The library adapter lazy-imports ``bof_tpt.engine.PredictionEngine``; we
inject a stub module via ``sys.modules`` so the test does not require
Brian's package to be installed.

The REST adapter is exercised against a patched ``requests.Session.post``;
no real HTTP traffic is generated.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ``requests`` is only needed by the REST adapter tests. Skip those if the
# package is unavailable; Library + Protocol tests must still run.
try:
   import requests  # type: ignore
   _HAS_REQUESTS = True
except ImportError:
   requests = None  # type: ignore
   _HAS_REQUESTS = False

requires_requests = pytest.mark.skipif(
   not _HAS_REQUESTS, reason="requests not installed"
)

from drone_sim.prediction.bof_adapter import (
   BoFAdapter,
   BoFLibraryAdapter,
   BoFRestAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubEngine:
   """Stub for bof_tpt.engine.PredictionEngine.

   Mirrors the new contract:
       PredictionEngine.from_config(horizon=H, config_path=None)
       engine.predict(history, has_velocity=False, drone_id=0,
                      horizon=None) -> {"trajectory": ndarray, "radii": ndarray}

   Records constructor inputs and the last predict() call so tests can
   assert horizon and drone_id propagation.
   """

   def __init__(self, horizon: int):
      self.horizon = horizon
      self.last_history = None
      self.last_has_velocity = None
      self.last_drone_id = None
      self.last_growth_tau = "<unset>"  # sentinel to distinguish None from not-passed

   @classmethod
   def from_config(cls, horizon: int, config_path=None):
      return cls(horizon=int(horizon))

   def predict(self, history: np.ndarray, has_velocity: bool = False,
               drone_id: int = 0, horizon=None, growth_tau=None):
      self.last_history = np.asarray(history, dtype=float)
      self.last_has_velocity = has_velocity
      self.last_drone_id = int(drone_id)
      self.last_growth_tau = growth_tau
      cols = 6 if has_velocity else 3
      h = int(horizon) if horizon is not None else self.horizon
      traj = np.full((h, cols), 0.7)
      radii = np.full(h, 1.3)
      return {"trajectory": traj, "radii": radii}


@pytest.fixture
def stub_bof_module(monkeypatch):
   """Inject a fake ``bof_tpt.engine`` into sys.modules.

   Keeps any pre-existing real ``bof_tpt`` modules intact for the duration
   of the test; only ``bof_tpt.engine`` is swapped for the stub.
   """
   stub_engine_mod = types.ModuleType("bof_tpt.engine")
   stub_engine_mod.PredictionEngine = _StubEngine

   if "bof_tpt" not in sys.modules:
      stub_pkg = types.ModuleType("bof_tpt")
      stub_pkg.engine = stub_engine_mod
      monkeypatch.setitem(sys.modules, "bof_tpt", stub_pkg)
   monkeypatch.setitem(sys.modules, "bof_tpt.engine", stub_engine_mod)
   yield stub_engine_mod


# ---------------------------------------------------------------------------
# BoFLibraryAdapter
# ---------------------------------------------------------------------------

class TestBoFLibraryAdapter:

   def test_constructor_passes_horizon_to_engine(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=4)
      assert adapter._predictor.horizon == 4

   def test_predict_default_has_velocity_false_slices_to_3_cols(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3)
      history = np.random.randn(10, 6)
      traj, radii = adapter.predict(history)

      assert adapter._predictor.last_has_velocity is False
      assert adapter._predictor.last_history.shape == (10, 3)
      np.testing.assert_array_equal(adapter._predictor.last_history, history[:, :3])
      assert traj.shape == (3, 3)
      assert radii.shape == (3,)

   def test_predict_has_velocity_true_sends_6_cols_and_slices_response(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3, has_velocity=True)
      history = np.random.randn(10, 6)
      traj, radii = adapter.predict(history)

      assert adapter._predictor.last_has_velocity is True
      assert adapter._predictor.last_history.shape == (10, 6)
      assert traj.shape == (3, 3)  # adapter slices the (3, 6) stub response

   def test_predict_forwards_drone_id(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3)
      adapter.predict(np.zeros((10, 6)), drone_id=7)
      assert adapter._predictor.last_drone_id == 7

   def test_predict_drone_id_none_defaults_to_zero(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3)
      adapter.predict(np.zeros((10, 6)))
      assert adapter._predictor.last_drone_id == 0

   def test_growth_tau_none_by_default(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3)
      adapter.predict(np.zeros((10, 6)))
      assert adapter._predictor.last_growth_tau is None

   def test_growth_tau_forwarded(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3, growth_tau=12.5)
      adapter.predict(np.zeros((10, 6)))
      assert adapter._predictor.last_growth_tau == 12.5

   def test_predict_short_history_columns_raises(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3, has_velocity=True)
      with pytest.raises(ValueError, match="needs 6"):
         adapter.predict(np.random.randn(10, 3))

   def test_lazy_import_failure_only_on_construction(self, monkeypatch):
      monkeypatch.setitem(sys.modules, "bof_tpt.engine", None)
      with pytest.raises((ImportError, TypeError, AttributeError)):
         BoFLibraryAdapter(horizon=3)


# ---------------------------------------------------------------------------
# BoFRestAdapter
# ---------------------------------------------------------------------------

def _mk_response(json_body: dict, status: int = 200) -> MagicMock:
   resp = MagicMock()
   resp.status_code = status
   resp.json.return_value = json_body
   resp.headers = {"X-BoF-Version": "test"}
   if status >= 400:
      resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
   else:
      resp.raise_for_status.return_value = None
   return resp


@requires_requests
class TestBoFRestAdapter:

   def test_predict_posts_to_api_predict_with_drone_id(self):
      adapter = BoFRestAdapter(
         url="http://localhost:5003/", horizon=4, timeout_s=1.5,
      )
      history = np.arange(60, dtype=float).reshape(10, 6)
      response_body = {
         "status": "OK",
         "active_model": "default",
         "trajectory": [[0.1, 0.2, 0.3]] * 4,
         "radii": [1.1, 1.2, 1.3, 1.4],
      }
      with patch.object(adapter._session, "post", return_value=_mk_response(response_body)) as mock_post:
         traj, radii = adapter.predict(history, drone_id=2)

      mock_post.assert_called_once()
      call_args = mock_post.call_args
      assert call_args.args[0] == "http://localhost:5003/api/predict"
      assert call_args.kwargs["timeout"] == 1.5
      payload = call_args.kwargs["json"]
      assert payload["horizon"] == 4
      assert payload["has_velocity"] is False
      assert payload["drone_id"] == 2
      assert payload["history"] == history[:, :3].tolist()

      assert traj.shape == (4, 3)
      np.testing.assert_array_almost_equal(traj, np.full((4, 3), [0.1, 0.2, 0.3]))
      np.testing.assert_array_almost_equal(radii, [1.1, 1.2, 1.3, 1.4])

   def test_has_velocity_true_sends_6_cols_and_slices_response(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3, has_velocity=True)
      history = np.arange(60, dtype=float).reshape(10, 6)
      response_body = {
         "status": "OK",
         "trajectory": [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]] * 3,
         "radii": [0.5, 0.5, 0.5],
      }
      with patch.object(adapter._session, "post", return_value=_mk_response(response_body)) as mock_post:
         traj, _ = adapter.predict(history)

      payload = mock_post.call_args.kwargs["json"]
      assert payload["has_velocity"] is True
      assert payload["history"] == history.tolist()
      assert traj.shape == (3, 3)

   def test_url_trailing_slash_is_stripped(self):
      adapter = BoFRestAdapter(url="http://x:1/", horizon=2)
      assert adapter._endpoint == "http://x:1/api/predict"
      adapter2 = BoFRestAdapter(url="http://x:1", horizon=2)
      assert adapter2._endpoint == "http://x:1/api/predict"

   def test_http_400_bad_request_raises(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      with patch.object(adapter._session, "post", return_value=_mk_response({}, status=400)):
         with pytest.raises(requests.HTTPError):
            adapter.predict(np.zeros((10, 6)))

   def test_http_422_fit_failed_raises(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      with patch.object(adapter._session, "post", return_value=_mk_response({}, status=422)):
         with pytest.raises(requests.HTTPError):
            adapter.predict(np.zeros((10, 6)))

   def test_http_500_raises(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      with patch.object(adapter._session, "post", return_value=_mk_response({}, status=500)):
         with pytest.raises(requests.HTTPError):
            adapter.predict(np.zeros((10, 6)))

   def test_timeout_propagates(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      with patch.object(adapter._session, "post", side_effect=requests.Timeout("slow")):
         with pytest.raises(requests.Timeout):
            adapter.predict(np.zeros((10, 6)))

   def test_connection_error_propagates(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      with patch.object(adapter._session, "post", side_effect=requests.ConnectionError("nope")):
         with pytest.raises(requests.ConnectionError):
            adapter.predict(np.zeros((10, 6)))

   def test_drone_id_none_defaults_to_zero(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      response_body = {
         "status": "OK",
         "trajectory": [[0.0, 0.0, 0.0]] * 3,
         "radii": [0.1, 0.1, 0.1],
      }
      with patch.object(adapter._session, "post", return_value=_mk_response(response_body)) as mock_post:
         adapter.predict(np.zeros((10, 6)))

      assert mock_post.call_args.kwargs["json"]["drone_id"] == 0

   def test_growth_tau_in_payload_default_none(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      response_body = {"status": "OK", "trajectory": [[0.0, 0.0, 0.0]] * 3, "radii": [0.1, 0.1, 0.1]}
      with patch.object(adapter._session, "post", return_value=_mk_response(response_body)) as mock_post:
         adapter.predict(np.zeros((10, 6)))
      assert mock_post.call_args.kwargs["json"]["growth_tau"] is None

   def test_growth_tau_in_payload_when_set(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3, growth_tau=8.0)
      response_body = {"status": "OK", "trajectory": [[0.0, 0.0, 0.0]] * 3, "radii": [0.1, 0.1, 0.1]}
      with patch.object(adapter._session, "post", return_value=_mk_response(response_body)) as mock_post:
         adapter.predict(np.zeros((10, 6)))
      assert mock_post.call_args.kwargs["json"]["growth_tau"] == 8.0


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:

   def test_library_adapter_satisfies_protocol(self, stub_bof_module):
      adapter = BoFLibraryAdapter(horizon=3)
      assert isinstance(adapter, BoFAdapter)

   @requires_requests
   def test_rest_adapter_satisfies_protocol(self):
      adapter = BoFRestAdapter(url="http://localhost:5003", horizon=3)
      assert isinstance(adapter, BoFAdapter)
