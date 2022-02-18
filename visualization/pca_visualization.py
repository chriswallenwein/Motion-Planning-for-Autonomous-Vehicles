import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class PCAVisualization():
  def __init__(self, pca_dir: str, heatmap_dir: str, scenario_ground_truth, hidden_representation):
      self._pca_model = joblib.load(pca_dir)
      self._hidden_representation = hidden_representation
      self._scenario_ground_truth = scenario_ground_truth
      
      x, y, color = self._load_pca(heatmap_dir)
      sorted_indices = color.argsort()
      self._x, self._y, self._color = x[sorted_indices], y[sorted_indices], color[sorted_indices]

  def visualize(self, timestep: int, ego_vehicle_id: int = None, color="#64FF64"):
    fig, ax = plt.subplots()
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)

    img = plt.scatter(self._x, self._y, c=self._color, s=0.5, edgecolors="none",cmap="inferno")
    cbar = plt.colorbar(img, ax=ax)

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Probability", rotation=270)

    if ego_vehicle_id:
      assert timestep >= 0, "Please specify a positive timestep."
      assert timestep < len(self._hidden_representation), "This timestep is not in the scenario."
      assert ego_vehicle_id in self._scenario_ground_truth[timestep].obstacle_ids, "This ego vehicle id is not in the scenario."
      
      hidden_representation = self._hidden_representation[timestep].detach().numpy()
      ego_vehicle_row = (self._scenario_ground_truth[timestep].obstacle_ids == ego_vehicle_id).nonzero(as_tuple=True)[0]
      ego_vehicle_representation = np.expand_dims(hidden_representation[ego_vehicle_row],axis=0)
      pca_out = self._pca_model.transform(ego_vehicle_representation)
      pca1 = pca_out[0][0]
      pca2 = pca_out[0][1]
      plt.scatter(pca1,pca2,s=20, c=[color])
      
    plt.grid()
    plt.show()
    return fig

  def _load_pca(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    file = open(filename, 'rb')
    r = pickle.load(file)
    file.close()
    return r