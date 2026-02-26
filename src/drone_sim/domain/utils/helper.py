from drone_sim.domain.drone import Drone

def all_drones_reached_destination(drones: list[Drone], thresh: float = 0.1) -> bool:
   reached = [drone.route.target_reached(position=drone.position(), thresh=thresh) for drone in drones]
   return all(reached)
