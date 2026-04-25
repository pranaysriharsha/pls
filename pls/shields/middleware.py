import torch
import os

class Middleware:
    """
    Differentiable Middleware for Probabilistic Logic Shielding.
    Reads safety thresholds from a configuration txt file and computes soft continuous safety score sigma.
    """
    def __init__(self, shield_layer, num_actions, config_folder, thresholds_file, k=10.0):
        self.shield_layer = shield_layer
        self.num_actions = num_actions
        self.k = k
        self.risk_thresholds = {}
        print("middleware initialsing")

        if thresholds_file is None:
            # Default fallback if no file provided
            self.risk_thresholds = {"unsafe_next": 0.5}
        else:
            print("thresholds file is ", thresholds_file)
            filepath = os.path.join(config_folder, thresholds_file)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Threshold file not found: {filepath}")
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":")
                    if len(parts) == 2:
                        atom = parts[0].strip()
                        threshold = float(parts[1].strip())
                        self.risk_thresholds[atom] = threshold

    def compute_sigma(self, risk_atom_name, current_prob):
        """
        Calculates the safety score sigma for a given risk atom.
        """
        threshold = self.risk_thresholds[risk_atom_name]
        return torch.sigmoid(self.k * (threshold - current_prob))

    def get_action_safeties(self, sensor_values, prev_sensors, prev_actions):
        """
        Calculates safety scores for all actions based on risk atoms.
        """
        all_actions = torch.eye(self.num_actions).unsqueeze(1)
        risk_atom_safeties = {atom: [] for atom in self.risk_thresholds.keys()}
        
        for action in all_actions:
            base_actions = torch.repeat_interleave(action, sensor_values.size(0), dim=0)
            results = self.shield_layer(
                x={
                    "sensor_value": sensor_values,
                    "action": base_actions,
                    "prev_sensor": prev_sensors,
                    "prev_action": prev_actions,
                }
            )
            for atom in risk_atom_safeties.keys():
                risk_atom_safeties[atom].append(results[atom])
                
        sigma_totals = None
        for atom, safeties in risk_atom_safeties.items():
            raw_probs = torch.cat(safeties, dim=1)
            sigma_i = self.compute_sigma(atom, raw_probs)
            if sigma_totals is None:
                sigma_totals = sigma_i
            else:
                sigma_totals = sigma_totals * sigma_i
                
        return sigma_totals
