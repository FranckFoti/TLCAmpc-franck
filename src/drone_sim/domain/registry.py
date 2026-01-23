from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Factory(Protocol[T]):
   def __call__(self, **kwargs: Any) -> T: ...


PHYSICS: dict[str, Factory[Any]] = {}
CONTROLLERS: dict[str, Factory[Any]] = {}
COORDINATORS: dict[str, Factory[Any]] = {}


def register_physics(name: str) -> Callable[[Factory[Any]], Factory[Any]]:
   def _decorator(factory: Factory[Any]) -> Factory[Any]:
      PHYSICS[name] = factory
      return factory

   return _decorator


def register_controller(name: str) -> Callable[[Factory[Any]], Factory[Any]]:
   def _decorator(factory: Factory[Any]) -> Factory[Any]:
      CONTROLLERS[name] = factory
      return factory

   return _decorator


def register_coordinator(name: str) -> Callable[[Factory[Any]], Factory[Any]]:
   def _decorator(factory: Factory[Any]) -> Factory[Any]:
      COORDINATORS[name] = factory
      return factory

   return _decorator


def create_from_registry(spec: dict[str, Any], registry: dict[str, Factory[Any]], kind: str) -> Any:
   t = spec.get("type")
   if not t:
      raise ValueError(f"{kind} spec must include 'type'")
   if t not in registry:
      raise ValueError(f"Unknown {kind} type: {t}")
   params = dict(spec.get("params") or {})
   return registry[t](**params)


def create_physics(spec: dict[str, Any]) -> Any:
   return create_from_registry(spec, PHYSICS, "physics")


def create_controller(spec: dict[str, Any]) -> Any:
   return create_from_registry(spec, CONTROLLERS, "controller")


def create_coordinator(spec: dict[str, Any]) -> Any:
   return create_from_registry(spec, COORDINATORS, "coordinator")
