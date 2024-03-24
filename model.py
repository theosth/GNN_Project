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
        self, num_series, num_timesteps, num_layers=2, hidden_size=128, dropout_rate=0.5
    ):
        super(ImprovedParameterPredictor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=num_series, out_channels=hidden_size, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size * 2,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        # Adjust the linear layer input size according to the convolution and pooling layers output
        pooled_sequence_length = num_timesteps // 2
        
        self.fc = nn.Linear(
            hidden_size * pooled_sequence_length, 11
        )  # Example output size adjustment

    def _transform_input_shape(self, x):
        # transform input of shape: (batch_size, num_channels, sequence_length)
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
        x = self.fc(x)
        return x


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


if __name__ == "__main__":

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
    BATCH_SIZE = 32

    # Generate training data
    labels = ["num_bodies", "masses", "radii", "acceleration_coefficients", "initial_v"]
    prior = Prior(prior_fn_basic, labels)

    X_train, Y_train = generate_training_data(
        NUM_TRAIN_SAMPLES, VARIABLES, total_time, dt, prior
    )
    # Generate test data
    X_test, Y_test = generate_training_data(
        NUM_TEST_SAMPLES, VARIABLES, total_time, dt, prior
    )

    # Create a dataset and dataloader
    train_dataset = MyDataset(X_train, Y_train)
    test_dataset = MyDataset(X_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # model = ParameterPredictor(
    #     input_size=TIME_STEPS*NUM_BODIES*PARAMS_PER_TIME_STEP_PER_BODY,
    #     hidden_size=128,
    #     num_l_layers=4,
    #     output_size=11
    # )

    model = ImprovedParameterPredictor(
        num_series=NUM_BODIES * PARAMS_PER_TIME_STEP_PER_BODY,  # datapoints per timestep
        num_timesteps=TIME_STEPS,
        num_layers=2,  # number of convolutional layers
        hidden_size=128,  # size of the hidden layers in fc part
        dropout_rate=0.5,  # dropout rate for regularization
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # output_size = 3 (number of bodies, constant mass, constant radius)
    # + NUM_BODIES * 4 (initial_velocity_x, initial_velocity_y, initial_position_x, initial_position_y)

    num_epochs = 100  # Example number of epochs
    learning_rate = 0.001  # Example learning rate

    train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate)
