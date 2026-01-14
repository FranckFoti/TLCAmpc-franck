from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Factory(Protocol[T]):
   def __call__(self, **kwargs: Any) -> T: ...


_PHYSICS: dict[str, Factory[Any]] = {}
_CONTROLLERS: dict[str, Factory[Any]] = {}
_COORDINATORS: dict[str, Factory[Any]] = {}


def register_physics(name: str) -> Callable[[Factory[Any]], Factory[Any]]:
   def _decorator(factory: Factory[Any]) -> Factory[Any]:
      _PHYSICS[name] = factory
      return factory

   return _decorator


def register_controller(name: str) -> Callable[[Factory[Any]], Factory[Any]]:
   def _decorator(factory: Factory[Any]) -> Factory[Any]:
      _CONTROLLERS[name] = factory
      return factory

   return _decorator


def register_coordinator(name: str) -> Callable[[Factory[Any]], Factory[Any]]:
   def _decorator(factory: Factory[Any]) -> Factory[Any]:
      _COORDINATORS[name] = factory
      return factory

   return _decorator


def create_physics(spec: dict[str, Any]) -> Any:
   t = spec.get("type")
   if not t:
      raise ValueError("physics spec must include 'type'")
   if t not in _PHYSICS:
      raise ValueError(f"Unknown physics type: {t}")
   params = dict(spec.get("params") or {})
   return _PHYSICS[t](**params)


def create_controller(spec: dict[str, Any]) -> Any:
   t = spec.get("type")
   if not t:
      raise ValueError("controller spec must include 'type'")
   if t not in _CONTROLLERS:
      raise ValueError(f"Unknown controller type: {t}")
   params = dict(spec.get("params") or {})
   return _CONTROLLERS[t](**params)


def create_coordinator(spec: dict[str, Any]) -> Any:
   t = spec.get("type")
   if not t:
      raise ValueError("coordinator spec must include 'type'")
   if t not in _COORDINATORS:
      raise ValueError(f"Unknown coordinator type: {t}")
   params = dict(spec.get("params") or {})
   return _COORDINATORS[t](**params)
