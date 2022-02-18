from PIL import Image
from typing import List
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle

class BaseRenderer():
  # the base renderer class is responsible for drawing everything regarding scenarios
  def __init__(self):
    pass

  def draw_lanelets(self, canvas: Image.Image, lanelets: List[Lanelet]):
    for lanelet in lanelets:
      poly_path = lanelet.polygon.vertices
      canvas.draw_polygon(poly_path)
    return canvas

  def draw_predictions(self, canvas: Image.Image, timestep: int, ego_vehicle_id: int, predictions, graphs, scenario,color = (100,255,100,255)):
    if timestep < len(graphs):
      predictions = predictions[timestep]
      ego_row = (graphs[timestep].obstacle_ids == ego_vehicle_id).nonzero()[0].item()
      for angular_region in range(8):
        closeness = predictions[ego_row, angular_region]
        self.draw_angular_region(canvas=canvas, timestep=timestep, ego_vehicle_id=ego_vehicle_id, angular_region=angular_region, scenario=scenario, closeness=closeness)
    return canvas

  def draw_angular_region(self, canvas: Image.Image, timestep: int, ego_vehicle_id: int, angular_region: int, scenario, closeness: float, color = (100,255,100,255)):
    state = scenario.obstacle_by_id(ego_vehicle_id).state_at_time(timestep)
    
    # Obstacle not present at this time-step
    if state is None: return canvas

    ego_position = state.position[0].item(), state.position[1].item()
    distance = self.closeness_to_distance(closeness)    
    bounding_box = [((ego_position[0] - distance), (ego_position[1] - distance)), ((ego_position[0] + distance), (ego_position[1] + distance))]
    angles = self.angular_region_to_angles(angular_region)
          
    canvas.draw_pieslice(bounding_box, *angles, outline=color, width=1)
    return canvas

  def draw_vehicles(self, canvas: Image.Image, vehicles: List[DynamicObstacle], timestep: int, ego_vehicle_id: int):
    for vehicle in vehicles:
      state = vehicle.state_at_time(timestep)
      if state is None:
        # Obstacle not present at this time-step
        continue
      color = (255,100,100,255) if vehicle.obstacle_id == ego_vehicle_id else (100,100,255,255)

      origin = state.position
      relative_poly_path = vehicle.obstacle_shape.vertices[:-1]
      rotation = state.orientation
      canvas.draw_relative_polygon(origin, relative_poly_path, rotation, fill=color, outline=color)
    return canvas

  def closeness_to_distance(self, closeness: float) -> float :
    return (1-closeness)*200

  def angular_region_to_angles(self, angular_region: int):
    assert angular_region >= 0 and angular_region < 8, "angular region must be between 0 and 7"
    angular_sections = [(45, 90), (0, 45), (315, 360), (270, 315), (225, 270), (180, 225), (135, 180), (90, 135)]
    return angular_sections[angular_region]