from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from typing import List, Union, Tuple

class ScenarioUtil():
  
  @classmethod
  def get_last_timestep(cls, scenario: Scenario) -> Union[int, int]:
    return max([obstacle.prediction.final_time_step for obstacle in scenario.dynamic_obstacles])

  @classmethod
  def get_boundary(cls, lanelets: List[Lanelet], obstacles: List[DynamicObstacle]) -> Tuple[float, float, float, float]:
    def get_obstacle_boundary(obstacle: DynamicObstacle) -> Tuple[float, float, float, float]:
      final_timestep = obstacle.prediction.final_time_step
      states = [obstacle.state_at_time(timestep) for timestep in range(final_timestep+1)]
      states = [not_none for not_none in states if not_none]
      positions = [state.position for state in states]
      x_values = [position[0] for position in positions]
      y_values = [position[1] for position in positions]
      return min(x_values), max(x_values) , min(y_values), max(y_values)

    obstacle_boundaries = [get_obstacle_boundary(obstacle) for obstacle in obstacles]
    x_min, x_max, y_min, y_max = 0, 1, 2, 3
    obstacle_x_min = min([boundary[x_min] for boundary in obstacle_boundaries])
    obstacle_x_max = min([boundary[x_max] for boundary in obstacle_boundaries])
    obstacle_y_min = min([boundary[y_min] for boundary in obstacle_boundaries])
    obstacle_y_max = min([boundary[y_max] for boundary in obstacle_boundaries])

    start, end = 0, -1
    x, y = 0, 1
    lanelet_x_min = min([min(ll.left_vertices[start][x], ll.right_vertices[start][x]) for ll in lanelets])
    lanelet_x_max = max([max(ll.left_vertices[end][x], ll.right_vertices[end][x]) for ll in lanelets])
    lanelet_y_min = min([min(ll.left_vertices[start][y], ll.right_vertices[start][y]) for ll in lanelets])
    lanelet_y_max = max([max(ll.left_vertices[end][y], ll.right_vertices[end][y]) for ll in lanelets])

    return min(obstacle_x_min, lanelet_x_min), max(obstacle_x_max, lanelet_x_max), min(obstacle_y_min, lanelet_y_min), max(obstacle_y_max, lanelet_y_max)

  @classmethod
  def get_obstacle_ids_at_timestep(cls, scenario: Scenario, timestep: int):
    return scenario.obstacle_states_at_time_step(timestep).keys()