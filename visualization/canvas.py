from commonroad.scenario.scenario import Scenario
from util.scenario_util import ScenarioUtil
from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, List

class Canvas():
  # the canvas class is responsible for all things drawing
  def __init__(self, scenario: Scenario, scale: int = 2, border: int = 50):
    self._border = border
    self.x_min, self.x_max, self.y_min, self.y_max = ScenarioUtil.get_boundary(scenario.lanelet_network.lanelets, scenario.obstacles)
    self.x_min, self.x_max, self.y_min, self.y_max = self.x_min - self._border, self.x_max + self._border, self.y_min - self._border, self.y_max + self._border
    width = int((self.x_max - self.x_min)*scale)
    height = int((self.y_max - self.y_min)*scale)
    self._canvas = Image.new("RGBA", (width, height), color = (50,50,50,255))
    self._draw = ImageDraw.Draw(self._canvas)

  def zoom(self, scenario: Scenario, timestep: int, ego_vehicle_id: int, zoom: Tuple[int, int] = (250, 200)):
    
    ego_vehicle_position = scenario.obstacle_by_id(ego_vehicle_id).state_at_time(timestep).position.tolist()
    x, y = self.transform([tuple(ego_vehicle_position)])[0]
    
    x_range, y_range = zoom

    zoom_settings = (x - x_range, y - y_range, x + x_range, y + y_range)

    self._canvas = self._canvas.crop(zoom_settings)
    return self
    
  def show(self):
      return self._canvas.transpose(Image.FLIP_TOP_BOTTOM)

  def draw_polygon(self, poly_path, fill=None, outline=None):
    poly_path = self.transform(poly_path)
    self._draw.polygon(poly_path, fill=fill, outline=outline)
    return self

  def draw_relative_polygon(self, origin, relative_poly_path, rotation, scale = 1, fill=None, outline=None):
    # draw a polygon relative to origin
    def transform_vertex(origin, relative_point,rotation, scale):
      return (np.cos(rotation) * relative_point[0] * scale - np.sin(rotation) * relative_point[1] * scale + origin[0],
              np.sin(rotation) * relative_point[0] * scale + np.cos(rotation) * relative_point[1] * scale + origin[1])
      
    poly_path = [transform_vertex(origin, point, rotation, scale) for point in relative_poly_path]
    poly_path = self.transform(poly_path)
    self._draw.polygon(poly_path, fill=fill, outline=outline)
    return self

  def draw_pieslice(self, bounding_box, start_angle, end_angle, fill=None, outline=None, width:float = 1):
    bounding_box = self.transform(bounding_box)
    self._draw.pieslice(bounding_box, start_angle, end_angle, fill, outline, width)
    return self

  def transform(self, positions: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    # the canvas and the scenario have different coordinate systems
    # turn scenario coordinates into canvas coordinates
    canvas_x_max, canvas_y_max = self._canvas.size

    x_ratio = canvas_x_max / (self.x_max - self.x_min)
    y_ratio = canvas_y_max / (self.y_max - self.y_min)

    transform_x = lambda x: (x - self.x_min) * x_ratio
    transform_y = lambda y: (y - self.y_min) * y_ratio

    return [(transform_x(x), transform_y(y)) for x,y in positions]