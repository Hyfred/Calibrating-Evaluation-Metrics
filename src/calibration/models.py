import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from src.utils import set_seed


class NumpyDataset(Dataset):
    def __init__(self, features, labels, beta_values=None):
        self.features = features
        self.labels = labels
        self.beta_values = beta_values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.beta_values is not None:
            return self.features[idx], self.labels[idx], self.beta_values[idx]
        else:
            return self.features[idx], self.labels[idx]


class LogisticModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticModel, self).__init__()
        
        self.fixed_intercept = nn.Parameter(torch.randn(1))  # Intercept
        self.fixed_slope = nn.Linear(n_features, 1)  # Slope for individual predictor
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        # Compute logits: fixed intercept + fixed slope * X_individual
        logits = self.fixed_intercept + self.fixed_slope(h).squeeze()
        return self.sigmoid(logits)


def logistic_train(seed, h, y, l1_lambda=0.01, n_epochs=100, lr=0.05):
    set_seed(seed)
    h = torch.from_numpy(h).unsqueeze(-1).float()
    y = torch.from_numpy(y).float()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    device = "cpu"

    numpy_dataset = NumpyDataset(h, y)
    dataloader = DataLoader(numpy_dataset, batch_size=512, shuffle=True)

    model = LogisticModel(n_features=1).to(device)

    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    tolerance = 1e-4
    patience = 10
    patience_counter = 0
    for epoch in range(5):
        epoch_loss = 0.0
        model.train()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader) 
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
        if best_loss - epoch_loss < tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Training stopped early at epoch {epoch+1}. Loss plateaued.")
                break
        else:
            best_loss = epoch_loss
            patience_counter = 0  # Reset patience if loss improves

    return model


class MultilevelLogisticModel(nn.Module):
    def __init__(self, n_groups, n_features):
        super(MultilevelLogisticModel, self).__init__()
        self.fixed_intercept = nn.Parameter(torch.randn(1))

        self.fixed_slope = nn.Linear(n_features, 1)

        self.random_intercepts = nn.Embedding(n_groups, 1)

        self.random_slopes = nn.Embedding(n_groups, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X_individual, group_ids):        
        fixed_part = self.fixed_intercept + self.fixed_slope(X_individual).squeeze()
    
        # Identify which instances have NaN values in the input
        nan_mask = torch.isnan(group_ids).squeeze()

        # Initialize logits with only the fixed part
        logits = fixed_part.clone()
    
        if nan_mask.any():
            # Apply random intercept and slope for instances without NaN
            try:
                random_intercept = self.random_intercepts(
                    group_ids[~nan_mask].int()
                ).squeeze()
            except:
                import pdb

                pdb.set_trace()
            random_slope = (
                self.random_slopes(group_ids[~nan_mask].int()).squeeze()
                * X_individual[~nan_mask].squeeze()
            )
            
            logits[~nan_mask] += random_intercept + random_slope
        
        return self.sigmoid(logits)


def multilevel_logistic_train(
    seed, h, y, beta_values, l1_lambda=0.01, n_epochs=100, lr=0.05
):
    set_seed(seed)
    h = torch.from_numpy(h).unsqueeze(-1).float()
    y = torch.from_numpy(y).float()
    beta_values = torch.from_numpy(beta_values).float()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    numpy_dataset = NumpyDataset(h, y, beta_values)
    dataloader = DataLoader(numpy_dataset, batch_size=512, shuffle=True)

    # Initialize the model

    model = MultilevelLogisticModel(
        n_features=1, n_groups=torch.max(beta_values[~beta_values.isnan()]).int() + 1
    )

    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    tolerance = 1e-4
    patience = 10
    patience_counter = 0
    for epoch in range(5):
        epoch_loss = 0.0
        model.train()

        for inputs, targets, _beta_values in dataloader:
            inputs, targets, _beta_values = (
                inputs.to(device),
                targets.to(device),
                _beta_values.to(device),
            )
            optimizer.zero_grad() 
            outputs = model(inputs, _beta_values)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        
        if best_loss - epoch_loss < tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Training stopped early at epoch {epoch+1}. Loss plateaued.")
                break
        else:
            best_loss = epoch_loss
            patience_counter = 0  # Reset patience if loss improves

    return model


def forward_pass_pytorch(model, beta_values, confidence):
    model.eval() 
    
    with torch.no_grad():
        if beta_values is None:
            y_pred_probs = model(torch.from_numpy(confidence).unsqueeze(-1).float())
        else:
            y_pred_probs = model(
                torch.from_numpy(confidence).unsqueeze(-1).float(),
                torch.from_numpy(beta_values).float(),
            )

        y_pred = (y_pred_probs >= 0.5).float().squeeze()  # Get binary predictions

    return y_pred


def forward_pass_scikit(model, confidence):
    y_pred_probs = model.predict_proba(confidence.reshape(-1, 1))[:, 1]

    y_pred = (y_pred_probs >= 0.5).astype(int)  # Get binary predictions

    return y_pred


def print_train_accuracy(model, confidence, judgements, beta_values=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    try:
        # import pdb; pdb.set_trace()
        # beta_values = beta_values.to(device)
        # confidence = confidence.to(device)

        y_pred = forward_pass_pytorch(model, beta_values, confidence)
    except:
        y_pred = forward_pass_scikit(model, confidence)

    def compute_accuracy(y_pred, y_true):
        correct_predictions = (y_pred == y_true).sum()
        accuracy = correct_predictions / y_true.shape[0]
        return accuracy.item()

    accuracy = compute_accuracy(y_pred, judgements)
    print(f"Accuracy: {accuracy:.4f}")
