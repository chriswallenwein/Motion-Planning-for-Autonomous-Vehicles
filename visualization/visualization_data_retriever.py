from commonroad.common.file_reader import CommonRoadFileReader
import torch
from torch_geometric.data import Data
from model.traffic_representation_net import TrafficRepresentationNet, MLP, EdgeConv, EGATConvs
from extraction.extractor import TrafficDataExtractor
from commonroad_geometric_io.dataset.extraction.traffic.edge_drawers import FullyConnectedEdgeDrawer
from typing import List

class VisualizationDataRetriever:
  def __init__(self, scenario_path: str, model_path: str):
    self.scenario, _ = CommonRoadFileReader(filename=scenario_path).open()
    gnn = torch.load(model_path).cpu()
    self.graphs = self.get_graphs()
    self.predictions, self.hidden_representations = self.get_predictions(self.graphs, gnn)

  def get_graphs(self):
    return list(TrafficDataExtractor(
        scenario=self.scenario,
        edge_drawer=FullyConnectedEdgeDrawer(),
        render=False
        ))

  def get_predictions(self, graphs: List[Data], gnn):
    predictions = []
    hidden_representations = []
    for graph in graphs:
      graph = graph.cpu()
      
      x = graph.x
      edge_features = torch.zeros((graph.edge_attr.size(0), 17))
      edge_features[:, 5:8] = graph.edge_attr

      graph_connections = graph.edge_index

      classification_results, regression_results, classification_encoding, regression_encoding = gnn(x,graph_connections,edge_features)
      prediction = regression_results*((classification_results.gt(0.5).float()))
      hidden_representation = torch.cat((classification_encoding,regression_encoding),-1)

      predictions.append(prediction)
      hidden_representations.append(hidden_representation)

    return predictions, hidden_representations