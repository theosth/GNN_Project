from simulations.elastic_collisions import (
    Body,
    HiddenVariables,
    Variables,
    ElasticCollisionSimulation,
)
import torch
from torch import Tensor
from torch.distributions import Distribution
from typing import Union, Callable
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# prior and simulation interfaces
class Prior:
    def __init__(self, prior_fn, labels):
        self.prior_fn = prior_fn
        self.labels = labels
        self.mean = None
        self.std = None

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        samples = self.prior_fn(num_samples)
        self.mean = torch.mean(samples)
        self.std = torch.std(samples)
        return samples

    def get_stats(self):
        if self.mean is None and self.std is None:
            raise ValueError("Sample first before getting stats")
        else:
            return self.mean, self.std

    def get_labels(self):
        return self.labels


def prior_fn_basic(n_samples: int = 1) -> torch.Tensor:
    constant_mass_value = 1.0
    constant_radius_value = 1.0
    acceleration_coefficient_value = 0.0
    velocity_distribution = torch.distributions.Uniform(low=-4, high=4)

    num_bodies = torch.tensor(2)
    masses = torch.full((num_bodies,), constant_mass_value)
    radii = torch.full((num_bodies,), constant_radius_value)
    a_coeffs = torch.full((num_bodies,), acceleration_coefficient_value)
    initial_v = velocity_distribution.sample(sample_shape=torch.Size([num_bodies, 2]))

    flattened_tensors = torch.cat(
        [
            torch.flatten(num_bodies),
            torch.flatten(masses),
            torch.flatten(radii),
            torch.flatten(a_coeffs),
            torch.flatten(initial_v),
        ],
        dim=0,
    )

    # concatenate flattened tensors n_samples times
    return torch.stack([flattened_tensors for _ in range(n_samples)], dim=0)


def simulate_collisions_simple(Y_i, vars, total_time, dt):

    position_distribution = torch.distributions.Uniform(low=0.0, high=space_size)
    HIDDENVARIABLES = HiddenVariables.from_tensor(Y_i)

    initial_positions = (
        ElasticCollisionSimulation.sample_initial_positions_without_overlap(
            vars, position_distribution
        )
    )
    vars.starting_positions = initial_positions

    simulation = ElasticCollisionSimulation(
        variables=vars, enable_logging=False, noise=False
    )

    _ = simulation.simulate(
        hidden_variables=HIDDENVARIABLES, total_time=total_time, dt=dt
    )

    position_history = simulation.get_position_history()
    position_history_by_timestep_list = list(map(list, zip(*position_history)))
    t_tensor_list = [
        torch.stack(t_l).flatten() for t_l in position_history_by_timestep_list
    ]
    return torch.stack(t_tensor_list, dim=0)


def generate_training_data(num_samples, vars, total_time, dt, prior):
    Y = prior.sample(num_samples)
    Y_shape = Y.shape
    X = []
    for i in tqdm(range(num_samples)):
        Y_i = Y[i]
        Y_i_shape = Y_i.shape
        X_i = simulate_collisions_simple(Y_i, vars, total_time, dt)
        X_i_shape = X_i.shape
        # X.append(X_i.flatten())
        X.append(X_i)

    X_shape = X[0].shape
    X = torch.stack(X, dim=0)
    X_shape = X.shape
    return X, Y


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ParameterPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_l_layers: int = 1, output_size=11):
        super(ParameterPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1_s = self._build_fc_linear_layers(input_size, hidden_size, num_l_layers)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            hidden_size, output_size
        )  # Output layer with 11 neurons for the output values

    def _build_fc_linear_layers(self, input_size, hidden_size, num_layers):
        layers = []
        if num_layers == 0:
            raise ValueError("num_layers must be greater than 0")
        elif num_layers == 1:
            return [nn.Linear(input_size, hidden_size)]
        elif num_layers > 1:
            return [nn.Linear(input_size, hidden_size)] + [
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
            ]
        return layers

    def forward(self, x):
        # print(f"1: x shape: {x.shape}")
        x = self.flatten(x)
        # print(f"2: x shape: {x.shape}")
        for layer in self.fc1_s:
            temp_x = layer(x)
            # print(f"3: x shape: {x.shape}")
            x = self.relu(temp_x)
        # x = self.relu(x)
        x = self.fc2(x)
        return x


class SummaryNetwork(ParameterPredictor):
    def __init__(self, input_size, hidden_size, num_l_layers: int = 1):
        super().__init__(input_size, hidden_size, num_l_layers)
        # Remove the fc2 layer
        del self.fc2

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.fc1_s:
            x = self.relu(layer(x))
        return x


class ImprovedParameterPredictor(nn.Module):
    def __init__(
        self,
        num_series,
        num_timesteps,
        hidden_size=128,
        dropout_rate=0.1,
        fc_sizes=[256, 128],
        output_size=11,
    ):
        super(ImprovedParameterPredictor, self).__init__()
        self.num_series = num_series
        self.num_timesteps = num_timesteps
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.fc_sizes = fc_sizes
        self.output_size = output_size

        # set layers
        self.conv1 = nn.Conv1d(
            in_channels=self.num_series,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size * 2,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.dropout_rate)

        pooled_sequence_length = (
            self.num_timesteps // 4
        )  # div by 4 due to two pooling layers

        # caclulate the flattened size after the conv layers and pooling
        flattened_size = self.hidden_size * 2 * pooled_sequence_length

        # def fc layers
        fc_layers = []
        in_features = flattened_size
        for fc_size in self.fc_sizes:
            fc_layers.append(nn.Linear(in_features, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(self.dropout_rate))
            in_features = fc_size

        self.fc_layers = nn.ModuleList(fc_layers)
        self.linear = nn.Linear(in_features, self.output_size)

    def _transform_input_shape(self, x):
        # Transform input of shape: (batch_size, num_channels, sequence_length)
        # to shape: (batch_size, num_channels, sequence_length)
        return x.transpose(1, 2)

    def forward(self, x):
        x = self._transform_input_shape(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)

        for layer in self.fc_layers:
            x = layer(x)

        x = self.linear(x)
        return x


class ImprovedSummaryNetwork(ImprovedParameterPredictor):
    def __init__(
        self,
        num_series,
        num_timesteps,
        hidden_size,
        dropout_rate,
        fc_sizes,
    ):
        # Call the parent class constructor with all parameters except output_size
        super().__init__(
            num_series=num_series,
            num_timesteps=num_timesteps,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            fc_sizes=fc_sizes,
            output_size=1,  # some arbitrary value
        )
        # Remove the self.linear layer which is the last fully connected layer
        del self.linear

    @staticmethod
    def get_from_ImprovedParameterPredictor(base_model):
        summary_network = ImprovedSummaryNetwork(
            num_series=base_model.num_series,
            num_timesteps=base_model.num_timesteps,
            hidden_size=base_model.hidden_size,
            dropout_rate=base_model.dropout_rate,
            fc_sizes=base_model.fc_sizes,
        )

        # Copy the state_dict from the base model to the summary network
        # but ignore the self.linear layer's parameters
        base_state_dict = base_model.state_dict()
        summary_state_dict = {
            k: v for k, v in base_state_dict.items() if "linear" not in k
        }

        summary_network.load_state_dict(summary_state_dict)

        return summary_network

    def forward(self, x):
        """
        Forward pass through the network, skipping the last linear layer.
        """
        x = self._transform_input_shape(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)

        for layer in self.fc_layers:
            x = layer(x)

        # Skip the last fully connected layer as we've removed self.linear
        return x

    def get_last_layer_size(self):
        return self.fc_sizes[-1]


def train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_dataloader, leave=True, position=0)
        for inputs, labels in train_loop:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the tqdm progress bar for training
            train_loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
            train_loop.set_postfix(train_loss=loss.item())

        # Calculate and print average training loss
        avg_train_loss = running_loss / len(train_dataloader)
        # print(f"\nEpoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

        # Validation loss
        model.eval()
        total_val_loss = 0.0

        val_loop = tqdm(test_dataloader, leave=True, position=0)
        for inputs, labels in val_loop:
            with torch.no_grad():
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                val_loop.set_description("Validation")
                val_loop.set_postfix(val_loss=val_loss.item())

        # Calculate and print average validation loss
        avg_val_loss = total_val_loss / len(test_dataloader)
        # print(f"Average Validation Loss: {avg_val_loss}")

    print("Finished Training")


def subnet_constructor(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    return model


class CouplingBlock(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size):
        super(CouplingBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.split1 = math.floor(self.input_size / 2)
        self.split2 = self.input_size - self.split1
        self.subnet = subnet_constructor(
            self.split1 + self.condition_size, self.hidden_size, 2 * self.split2
        )

    def forward(self, x, cond):
        x1, x2 = x[..., : self.split1], x[..., self.split1 :]
        params = self.subnet(torch.cat([x1, cond], -1))
        s, t = params[..., : self.split2], params[..., self.split2 :]
        s = torch.tanh(s)
        ljd = torch.sum(s, -1)

        s = torch.exp(s)
        x2 = s * x2 + t
        return torch.cat([x1, x2], -1), ljd

    def inverse(self, y, cond):
        x1, x2 = y[..., : self.split1], y[..., self.split1 :]
        params = self.subnet(torch.cat([x1, cond], -1))
        s, t = params[..., : self.split2], params[..., self.split2 :]
        s = torch.tanh(s)
        ljd = torch.sum(s, -1)

        s = torch.exp(-s)
        x2 = s * (x2 - t)
        return torch.cat([x1, x2], -1)


class ConditionalRealNVP(nn.Module):
    def __init__(self, input_size, hidden_size, blocks, condition_size):
        super(ConditionalRealNVP, self).__init__()
        self.blocks = nn.ModuleList(
            [
                CouplingBlock(input_size, hidden_size, condition_size)
                for _ in range(blocks)
            ]
        )
        self.orthogonal_matrices = [
            torch.from_numpy(
                np.linalg.qr(np.random.randn(input_size, input_size))[0]
            ).float().to(device)
            for _ in range(blocks - 1)
        ]

    def forward(self, x, condition):
        log_det_jacobian = 0
        for i, block in enumerate(self.blocks):
            x, log_det_j = block(x, condition)
            log_det_jacobian += log_det_j
            if i != len(self.blocks) - 1:
                x = torch.matmul(
                    x, self.orthogonal_matrices[i]
                )  # changed from torch.mm to torch.matmul
        return x, log_det_jacobian

    def inverse(self, y, condition):
        for i, block in reversed(list(enumerate(self.blocks))):
            if i != len(self.blocks) - 1:
                y = torch.matmul(
                    y, self.orthogonal_matrices[i].inverse()
                )  # changed from torch.mm to torch.matmul
            y = block.inverse(y, condition)
        return y

    # def loss_function(self, z, log_det_jacobians):
    #     log_likelihood = -0.5 * torch.sum(z**2, dim=-1) - 0.5 * torch.log(
    #         torch.tensor(2 * np.pi).to(z.device)
    #     ) * z.size(
    #         1
    #     )  # changed from dim=1 to dim=-1
    #     return -(log_likelihood + log_det_jacobians).mean()

    def loss_function(self, z, log_det_jacobians):
        loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_jacobians)
        return loss

def train_INN(data_loader, inference_network, summary_network, learning_rate, train_config):
    noise = train_config.get("added_noise", 0.0)
    loss_history = {"train": [], "validation": []}
    epochs = train_config["epochs"]

    print("device: ", train_config["device"])

    optimizer = torch.optim.Adam(
        list(inference_network.parameters()) + list(summary_network.parameters()),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        inference_network.train()
        summary_network.train()
        total_loss = 0.0

        for sim_data, params_data in tqdm(
            data_loader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            sim_data = sim_data.to(train_config["device"])

            params_data = params_data.to(train_config["device"])

            # Add noise to the parameters data
            p_noise = params_data + torch.randn_like(params_data) * noise
            p_noise = p_noise.to(train_config["device"])
            # Forward pass
            summary_stats = summary_network(sim_data)

            # print(f"summary_stats: {summary_stats.shape}")
            # print(f"p_noise: {p_noise.shape}")
            z, log_det_jacobians = inference_network(p_noise, summary_stats)

            # Calculate loss
            loss = inference_network.loss_function(z, log_det_jacobians)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(data_loader)
        loss_history["train"].append(average_loss)

        # Validation loss
        with torch.no_grad():
            inference_network.eval()
            summary_network.eval()
            total_val_loss = 0.0
            for sim_data, params_data in data_loader:
                sim_data = sim_data.to(train_config["device"])
                params_data = params_data.to(train_config["device"])
                vp_noise = params_data + torch.randn_like(params_data) * noise
                v_summary_stats = summary_network(sim_data)
                v_z, v_log_det_jacobians = inference_network(vp_noise, v_summary_stats)
                v_loss = inference_network.loss_function(v_z, v_log_det_jacobians)
                total_val_loss += v_loss.item()
            average_val_loss = total_val_loss / len(data_loader)
            loss_history["validation"].append(average_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {average_loss}, Validation Loss: {average_val_loss}"
        )

    return loss_history


def get_posterior_samples(model, test_loader):
    model.eval()
    posterior_samples_list = []
    with torch.no_grad():
        for batch_idx, (X_test, Y_test) in enumerate(test_loader):  # Loop over batches
            X_test = X_test.to(train_config["device"])
            Y_test = Y_test.to(train_config["device"])

            summary_stats = summary_network(X_test)
            posterior_samples, _ = model(Y_test, summary_stats)

            # Convert posterior samples to numpy arrays and remove single-dimensional entries
            posterior_samples = np.squeeze(posterior_samples.cpu().numpy())

            # Reshape to flatten the batch dimension
            posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])

            posterior_samples_list.append(posterior_samples)

    # Combine all arrays in the list along the first axis
    return np.concatenate(posterior_samples_list, axis=0)

def plot_param_accuracy_scatter(predictions, ground_truth, file_path=None):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=100)
    axes = axes.flatten()

    for i in range(11):
        ax = sns.scatterplot(x=ground_truth[:, i], y=predictions[:, i], ax=axes[i], color="dodgerblue", edgecolor="w", s=100)
        min_val = min(ground_truth[:, i].min(), predictions[:, i].min()) - 0.1
        max_val = max(ground_truth[:, i].max(), predictions[:, i].max()) + 0.1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) # diagonal line
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f'Parameter {i+1}', fontsize=16)
        ax.set_xlabel('Ground Truth', fontsize=14)
        ax.set_ylabel('Prediction', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    axes[-1].set_visible(False)
    plt.tight_layout(pad=4.0)
    if file_path:
        plt.savefig(file_path)
    
    plt.show()
    
def plot_param_accuracy_box(predictions, ground_truth, file_path=None):
    deviations = predictions - ground_truth
    df_deviations = pd.DataFrame(deviations, columns=[f'Parameter {i+1}' for i in range(deviations.shape[1])])
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_deviations, palette="coolwarm")
    plt.title('Parameter Deviation Distribution', fontsize=18)
    plt.ylabel('Deviation', fontsize=14)
    plt.xlabel('Parameter', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_time = 10.0
    dt = 0.1
    NUM_BODIES = 2
    TIME_STEPS = int(total_time / dt)
    PARAMS_PER_TIME_STEP_PER_BODY = 2
    space_size = 10.0
    max_radius = space_size // 10.0
    acceleration_coefficient_value = 0.0
    constant_mass_value = 1.0
    constant_radius_value = max_radius

    VARIABLES = Variables(
        masses=torch.full((NUM_BODIES,), constant_mass_value),
        radii=torch.full((NUM_BODIES,), constant_radius_value),
        starting_positions=None,
        initial_velocities=None,
        acceleration_coefficients=torch.full(
            (NUM_BODIES,), acceleration_coefficient_value
        ),
        num_bodies=NUM_BODIES,
        space_size=torch.tensor([space_size, space_size]),
    )

    NUM_TRAIN_SAMPLES = 10000
    NUM_TEST_SAMPLES = 2000
    BATCH_SIZE = 64

    # Generate training data
    labels = ["num_bodies", "masses", "radii", "acceleration_coefficients", "initial_v"]
    prior = Prior(prior_fn_basic, labels)

    load_data = True
    if not load_data:
        X_train, Y_train = generate_training_data(
            NUM_TRAIN_SAMPLES, VARIABLES, total_time, dt, prior
        )
        # Generate test data
        X_test, Y_test = generate_training_data(
            NUM_TEST_SAMPLES, VARIABLES, total_time, dt, prior
        )
        # save data
        torch.save(X_train, "X_train.pt")
        torch.save(Y_train, "Y_train.pt")
        torch.save(X_test, "X_test.pt")
        torch.save(Y_test, "Y_test.pt")
    else:
        # load
        X_train = torch.load("X_train.pt")
        Y_train = torch.load("Y_train.pt")
        X_test = torch.load("X_test.pt")
        Y_test = torch.load("Y_test.pt")

    # move to GPU if available
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Create a dataset and dataloader
    train_dataset = MyDataset(X_train, Y_train)
    test_dataset = MyDataset(X_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model = ParameterPredictor(
    #     input_size=TIME_STEPS*NUM_BODIES*PARAMS_PER_TIME_STEP_PER_BODY,
    #     hidden_size=128,
    #     num_l_layers=4,
    #     output_size=11
    # )

    improved_parameter_predictor = ImprovedParameterPredictor(
        num_series=NUM_BODIES
        * PARAMS_PER_TIME_STEP_PER_BODY,  # datapoints per timestep
        num_timesteps=TIME_STEPS,
        hidden_size=128,  # size of the hidden layers in fc part
        dropout_rate=0.05,  # dropout rate for regularization
        fc_sizes=[256, 128, 128, 128, 64],  # size of the fully connected layers
    )
    improved_parameter_predictor.to(device)

    # output_size = 3 (number of bodies, constant mass, constant radius)
    # + NUM_BODIES * 4 (initial_velocity_x, initial_velocity_y, initial_position_x, initial_position_y)

    num_epochs = 100  # Example number of epochs
    learning_rate = 0.001  # Example learning rate

    train_model(
        improved_parameter_predictor,
        train_dataloader,
        test_dataloader,
        num_epochs,
        learning_rate,
    )

    summary_network = ImprovedSummaryNetwork.get_from_ImprovedParameterPredictor(
        improved_parameter_predictor
    )
    summary_network.to(device)

    # Create an instance of the RealNVP model
    summary_network_hidden_size = summary_network.get_last_layer_size()
    hidden_values_size = 11
    coupling_layers = 6
    realnvp_hidden_size = 128

    inference_network = ConditionalRealNVP(
        input_size=hidden_values_size,
        hidden_size=realnvp_hidden_size,
        blocks=coupling_layers,
        condition_size=summary_network_hidden_size,
    )
    inference_network.to(device)

    train_config = {"epochs": 20, "device": device, "added_noise": 0.0}

    loss_history = train_INN(
        data_loader=train_dataloader,
        inference_network=inference_network,
        summary_network=summary_network,
        train_config=train_config,
        learning_rate=0.001,
    )

    print("Training complete!")

    # plot results

    # get the posterior samples
    test_loader = DataLoader(MyDataset(X_test, Y_test), batch_size=1, shuffle=False)
    posterior_samples = get_posterior_samples(inference_network, test_loader)
    print(posterior_samples.shape)
    
    predictions = posterior_samples
    ground_truth = Y_test.cpu().numpy()
    
    # plot the data
    plot_param_accuracy_scatter(predictions, ground_truth, file_path="scatter_plot.png")

    plot_param_accuracy_box(predictions, ground_truth, file_path="box_plot.png")

