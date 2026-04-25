import torch as th
from pls.shields.deepproblog import DeepProbLogLayer, DeepProbLogLayer_Optimized
from pls.shields.middleware import Middleware
from os import path
from random import random


class Shield:
    """
    Shield structure for RL algorithms. It takes a policy and a set of observations to produce a safer policy
    using background knowledge. The structure is a compiled problog program. It is intended to be used on top of a policy network.

    Attributes
    ----------
    num_sensors: Number of sensors.
    num_actions: Number of available discrete actions.
    differentiable: Boolean indicating whether the shield is differentiable.
    config_folder: location of the config file
    observation_type:Observation type ("ground truth" or "pretrained")
    noisy_observations: Boolean indicating whether noisy observations are used
    shield_layer: deep problog layer as the shield
    observation_model: observation net object
    get_sensor_value_ground_truth: function used to compute ground truth observations from image input
    vsrl_eps: Probability of shutting down the shield.
    """

    def __init__(
        self,
        config_folder=None,
        get_sensor_value_ground_truth=None,
        net_input_dim=None,
        num_sensors=None,
        num_actions=None,
        shield_program=None,
        observation_type=None,
        noisy_observations=False,
        observation_net=None,
        differentiable=True,
        observation_net_cls=None,
        vsrl_eps=0,
        thresholds_file=None,
        k=10.0,
        **kwargs,
    ):
        if config_folder is None:
            # used only when loading an existing policy
            return

        self.num_sensors = num_sensors
        self.num_actions = num_actions
        self.differentiable = differentiable
        self.observation_type = observation_type
        with open(path.join(config_folder, shield_program)) as f:
            program = f.read()

        # Build Middleware first to read txt parameters securely
        self.middleware = Middleware(
            shield_layer=None, 
            num_actions=self.num_actions, 
            config_folder=config_folder,
            thresholds_file=thresholds_file,
            k=k
        )

        debug_query_struct = {atom: i for i, atom in enumerate(self.middleware.risk_thresholds.keys())}
        debug_input_struct = {
            "sensor_value": [i for i in range(self.num_sensors)],
            "action": [
                i for i in range(self.num_sensors, self.num_sensors + self.num_actions)
            ],
            "prev_sensor": [
                i for i in range(self.num_sensors + self.num_actions, 2 * self.num_sensors + self.num_actions)
            ],
            "prev_action": [
                i for i in range(2 * self.num_sensors + self.num_actions, 2 * self.num_sensors + 2 * self.num_actions)
            ],
        }

        self.shield_layer = self.get_layer(
            program=program,
            evidences=[],
            input_struct=debug_input_struct,
            query_struct=debug_query_struct,
        )
        
        # Hydrate backreference inside Middleware locally
        self.middleware.shield_layer = self.shield_layer

        # get sensor values from the pretrained observation network
        if self.observation_type == "pretrained":
            self.noisy_observations = noisy_observations
            use_cuda = False
            device = th.device("cuda" if use_cuda else "cpu")
            self.observation_model = observation_net_cls(
                input_size=net_input_dim*net_input_dim,
                output_size=self.num_sensors,
            ).to(device)
            observation_net_path = path.join(config_folder, observation_net)
            self.observation_model.load_state_dict(th.load(observation_net_path))
        elif self.observation_type == "ground truth":
            self.get_sensor_value_ground_truth = get_sensor_value_ground_truth

        # VSRL has a predefined parameter to randomize actions
        self.vsrl_eps = vsrl_eps

    def get_layer(self, program, evidences, input_struct, query_struct):
        """
        Initialize a problog layer.

        :param program: problog program
        :param queries: problog queries
        :param evidences: problog evidence
        :param input_struct: dict containing input values to the problog program
        :param query_struct: dict containing output
        :return: a problog layer
        """

        queries = query_struct.keys()
        # layer = DeepProbLogLayer_Optimized(
        #     program=program,
        #     queries=queries,
        #     evidences=evidences,
        #     input_struct=input_struct,
        #     query_struct=query_struct,
        # )
        layer = DeepProbLogLayer(
            program=program,
            queries=queries,
            evidences=evidences,
        )

        return layer

    def get_policy_safety(self, sensor_values, base_actions, prev_sensors, prev_actions) -> th.Tensor:
        """
        Calculates loss variables (expectation variables) if called natively.
        """
        #Calculates the action safeties
        sigma_totals = self.get_action_safeties(sensor_values, prev_sensors, prev_actions)
        #Calculates the shielded policy
        pi_plus_unnormalized = sigma_totals * base_actions
        sum_pi_plus = th.sum(pi_plus_unnormalized, dim=1, keepdim=True)
        pi_plus = pi_plus_unnormalized / (sum_pi_plus + 1e-8)
        return th.sum(pi_plus * sigma_totals, dim=1, keepdim=True)

    def get_action_safeties(self, sensor_values, prev_sensors, prev_actions) -> th.Tensor:
        """
        Compute how safe it is to execute an action. Let middleware execute it fully.
        """
        return self.middleware.get_action_safeties(sensor_values, prev_sensors, prev_actions)

    def get_shielded_policy(self, base_actions, sensor_values, prev_sensors, prev_actions) -> th.Tensor:
        """
        Compute the shielded policy statically from internally parsed sigmas securely.
        """
        assert self.differentiable is True

        action_safeties = self.get_action_safeties(sensor_values, prev_sensors, prev_actions)
        unnormalized_actions = action_safeties * base_actions
        actions = unnormalized_actions / (th.sum(unnormalized_actions, dim=1, keepdim=True) + 1e-8)

        assert actions.max() <= 1.00001, f"{actions} violates MAX"
        assert actions.min() >= -0.00001, f"{actions} violates MIN"

        return actions

    def get_shielded_policy_vsrl(self, base_actions, sensor_values, prev_sensors, prev_actions) -> th.Tensor:
        """
        Compute the shielded policy. This function is an implementation of vsrl
        (a non-differentiable shield).

        :param base_actions: tensor of the action probability distribution
        :param sensor_values: tensor of sensor values (observed or ground truth)
        :param prev_sensors: tensor of previous step sensor values
        :param prev_actions: tensor of previous step action one-hot vector
        :return: tensor representing the shielded policy
        """

        assert self.differentiable is False

        with th.no_grad():
            rdn = random()
            if rdn < self.vsrl_eps:
                # turn off the shield with the probability of self.vsrl_eps,
                # i.e. action_safeties contains only ones
                action_safeties = th.ones((sensor_values.size(0), self.num_actions))
            else:
                # turn on the shield
                action_safeties = self.get_action_safeties(sensor_values, prev_sensors, prev_actions)
                # vsrl requires all actions to be either safe or unsafe
                action_safeties = (action_safeties > 0.5).float()

            actions = (
                action_safeties
                * base_actions
                / th.sum(base_actions * action_safeties, dim=1, keepdim=True)
            )

            return actions

    def get_sensor_values(self, x: th.Tensor) -> th.Tensor:
        """
        Returns ground truth sensor values or ones from the observation network.

        :param x: tensor representing the states to extract features from
        :return: tensor of sensor values in [0, 1]
        """

        if self.observation_type == "pretrained":
            with th.no_grad():  # do not update the observation net
                if self.noisy_observations:
                    sensor_values = self.observation_model.get_sensor_values(x)
                else:
                    sensor_values = self.observation_model.get_sensor_values(x)
                    sensor_values = (sensor_values > 0.5).float()

        elif self.observation_type == "ground truth":
            with th.no_grad():  # do not update the observation net
                sensor_values = self.get_sensor_value_ground_truth(x)
        return sensor_values
