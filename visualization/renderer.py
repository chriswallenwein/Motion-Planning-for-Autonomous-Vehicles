from visualization.canvas import Canvas
from visualization.base_renderer import BaseRenderer

class Renderer():
  def __init__(self):
    pass

  def render(self, scenario, graphs, predictions, timestep, ego_vehicle_id, zoom):
    canvas = Canvas(scenario, scale=3)
    renderer = BaseRenderer()
    canvas = renderer.draw_lanelets(canvas, scenario.lanelet_network.lanelets)
    canvas = renderer.draw_vehicles(canvas, scenario.obstacles, timestep, ego_vehicle_id)
    canvas = renderer.draw_predictions(canvas=canvas, timestep=timestep, ego_vehicle_id=ego_vehicle_id, predictions=predictions, scenario=scenario, graphs=graphs)
    canvas = canvas.zoom(scenario, timestep, ego_vehicle_id, zoom)
    return canvas.show()

  def save_movie(self, scenario, graphs, predictions, ego_vehicle_id, filepath, zoom, frame_duration):
    ego_vehicle = scenario.obstacle_by_id(ego_vehicle_id)
    start, end = ego_vehicle.prediction.initial_time_step, ego_vehicle.prediction.final_time_step  
    images = [self.render(scenario, graphs, predictions, timestep, ego_vehicle_id, zoom) for timestep in range(start, end)]
    images[0].save(filepath, save_all=True, append_images=images, loop=0, duration=frame_duration)
    return filepath