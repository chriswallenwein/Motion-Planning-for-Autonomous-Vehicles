import torch
from torch_geometric.data import Data
from commonroad_geometric_io.dataset.extraction.traffic import StandardTrafficExtractor
import numpy as np


class TrafficDataExtractor(StandardTrafficExtractor):

    def num_node_features(self) -> int:
        return 7

    def num_edge_features(self) -> int:
        return 3

    def _extract(self, index: int) -> Data:

        """Creates graph representation of vehicles in a Commonroad scenario
        intended for PyTorch geometric. Uses Voronoi diagrams to decide which edges to draw.

        Args:
            t: Time-step of the scenario to process

        Returns:
            Data: Pytorch Geometric graph data instance.
        """
        timestep: int = self._get_time_step(index)
        if timestep >= len(self._vehicle_time_dict):
            return Data(x=torch.Tensor(), edge_index=torch.Tensor(), edge_attr=torch.Tensor(), y=torch.Tensor(), obstacle_ids=torch.Tensor)
        obstacles = self._vehicle_time_dict[timestep]
        n_vehicles: int = len(obstacles)
        if n_vehicles == 0:
            return Data(x=torch.Tensor(), edge_index=torch.Tensor(), edge_attr=torch.Tensor(), y=torch.Tensor(), obstacle_ids=torch.Tensor)
        n_edges: int = n_vehicles * (n_vehicles - 1)

        node_positions = self.get_node_positions(obstacles, timestep)

        return Data(x=node_positions,
            edge_index=self.get_fully_connected_edge_index(n_vehicles),
            edge_attr=self.calculate_edge_features(node_positions, n_edges),
            y=self.calculate_y(node_positions, n_vehicles),
            obstacle_ids=self.get_obstacle_ids(obstacles)
            )

    def calculate_y(self, node_positions: torch.Tensor, n_vehicles: int) -> torch.Tensor:
        # 1 means 0 metres
        # 0 means far away + we don't care

        y = torch.zeros((n_vehicles, 8))

        if n_vehicles == 1:
            return y

        for ego_vehicle_idx in range(n_vehicles):
            transformed_positions = self.transform_node_position(node_positions, ego_vehicle_idx)
            closeness = self.calculate_closeness(transformed_positions)

            # get the angular regions
            x_position = node_positions[:, 0]
            y_position = node_positions[:, 1]
            x_greater_0 = torch.greater(x_position, 0)
            y_greater_0 = torch.greater(y_position, 0)
            abs_x_greater_abs_y = torch.greater(torch.abs(x_position), torch.abs(y_position))

            region_0 = torch.eq(x_position, 0) * torch.eq(y_position, 0)
            region_1 = x_greater_0 * y_greater_0 * ~abs_x_greater_abs_y
            region_2 = x_greater_0 * y_greater_0 * abs_x_greater_abs_y
            region_3 = x_greater_0 * ~y_greater_0 * abs_x_greater_abs_y
            region_4 = x_greater_0 * ~y_greater_0 * ~abs_x_greater_abs_y
            region_5 = ~x_greater_0 * ~y_greater_0 * ~abs_x_greater_abs_y
            region_6 = ~x_greater_0 * ~y_greater_0 * abs_x_greater_abs_y
            region_7 = ~x_greater_0 * y_greater_0 * abs_x_greater_abs_y
            region_8 = ~x_greater_0 * y_greater_0 * ~abs_x_greater_abs_y

            y[ego_vehicle_idx, :] = torch.tensor([
                torch.max(closeness.where(region_1, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_2, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_3, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_4, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_5, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_6, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_7, torch.tensor([0], dtype=torch.double))),
                torch.max(closeness.where(region_8, torch.tensor([0], dtype=torch.double)))
            ])

        return y

    def transform_node_position(self, node_position: torch.Tensor, ego_vehicle_idx: int) -> torch.Tensor:
        return node_position - node_position[ego_vehicle_idx]

    def calculate_closeness(self, node_positions: torch.Tensor) -> torch.Tensor:
        distance = torch.sqrt(torch.sum(node_positions ** 2, 1))
        max_value = torch.empty(distance.size()).fill_(200)
        return 1 - torch.div(torch.minimum(distance, max_value), max_value)

    def get_node_positions(self, obstacles, timestep: int) -> torch.Tensor:
        # column | node feature
        # 0         x position
        # 1         y position
        return torch.vstack([torch.from_numpy(node.state_at_time(timestep).position) for node in obstacles])

    def calculate_edge_features(self, node_positions: torch.Tensor, n_edges: int) -> torch.Tensor:
        # column | edge feature
        # 0         euclidian distance between nodes
        # 1         sin(relative angle)
        # 2         cos(relative angle)

        edge_features = torch.zeros((n_edges, self.num_edge_features()), dtype=torch.float32)

        x_diff = (node_positions[:, 0].unsqueeze(1) - node_positions[:, 0].unsqueeze(0))
        x_diff = x_diff.masked_select(~torch.eye(node_positions.size(dim=0), dtype=torch.bool))

        y_diff = (node_positions[:, 1].unsqueeze(1) - node_positions[:, 1].unsqueeze(0))
        y_diff = y_diff.masked_select(~torch.eye(node_positions.size(dim=0), dtype=torch.bool))

        distance = torch.sqrt(x_diff ** 2 + y_diff ** 2)
        edge_features[:, 0] = distance
        edge_features[:, 1] = x_diff / distance
        edge_features[:, 2] = y_diff / distance
        return edge_features

    def get_obstacle_ids(self, obstacles) -> torch.Tensor:
        return torch.from_numpy(np.vectorize(lambda x: x.obstacle_id)(obstacles))

    def get_fully_connected_edge_index(self, n_vehicles: int) -> torch.Tensor:
        return (torch.ones((n_vehicles, n_vehicles)) - torch.eye(n_vehicles)).nonzero().T