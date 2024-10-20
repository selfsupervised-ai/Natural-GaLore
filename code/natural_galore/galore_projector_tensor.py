import torch
from tensorly.decomposition import tucker
from tensorly import tenalg

# Ensure tensorly uses PyTorch as the backend
import tensorly as tl

tl.set_backend("pytorch")


# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html
class GaLoreProjectorTensor:
    """
    GaLore Projector for tensors using tensor decomposition.

    Args:
        rank (int): The rank for the Tucker decomposition.
        update_proj_gap (int, optional): Iterations between updating the projection factors. Defaults to 200.
        scale (float, optional): Scaling factor for the projected gradients. Defaults to 1.0.
        proj_type (str, optional): Type of projection ('std' or 'continuous'). Defaults to 'std'.
    """

    def __init__(self, rank, update_proj_gap=200, scale=1.0, natural_history = 20, proj_type="galore"):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.core = None
        self.factors = None
        self.optimizers = None  # For continuous projection
        self.transformed_low_rank = None

        # Parameters for natural gradient approximation
        self.grad_history = []  # Buffer to store previous low-rank gradients
        self.history_size = natural_history  # Number of previous gradients to keep
        self.lambda_damping = 1e-3  # Damping term λ for numerical stability
        self.F_inv = None  # Inverse Fisher Information Matrix

    def project(self, full_rank_grad, iter, lr_ratio=1.0):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.
            lr_ratio (float): Ratio of current learning rate to initial learning rate.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        match self.proj_type:
            case "galore":
                if self.core is None or iter % self.update_proj_gap == 0:
                    self.core, self.factors = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank
                    )
                self.transformed_low_rank = self.transform(self.factors, full_rank_grad)
            case "natural_galore":
                if self.core is None or iter % self.update_proj_gap == 0:
                    self.core, self.factors = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank
                    )
                self.transformed_low_rank = self.transform(self.factors, full_rank_grad)
                self.transformed_low_rank = self._natural_gradient_transform(
                    self.transformed_low_rank
                )
            case "online_galore":
                self.transformed_low_rank = self._project_online_galore(
                    full_rank_grad, lr_ratio
                )
            case "online_natural_galore":
                self.transformed_low_rank = self._project_online_galore(
                    full_rank_grad, lr_ratio
                )
                self.transformed_low_rank = self._natural_gradient_transform(
                    self.transformed_low_rank
                )
            case _:
                raise NotImplementedError(
                    f"Projection type '{self.proj_type}' is not implemented."
                )
        return self.transformed_low_rank

    def _project_online_galore(self, full_rank_grad, lr_ratio):
        """
        Projects the full-rank gradients onto the low-rank subspace using online GaLore.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            lr_ratio (float): Ratio of current learning rate to initial learning rate.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.core is None:
            self.core, self.factors = self.get_orthogonal_matrix(
                full_rank_grad, self.rank
            )
            # Initialize optimizers for factors
            self.optimizers = []
            for factor in self.factors:
                factor.requires_grad = True
                optimizer = torch.optim.AdamW([factor], lr=1.0 / self.update_proj_gap)
                self.optimizers.append(optimizer)
        else:
            self._update_factors(full_rank_grad, lr_ratio)
        return self.transform(self.factors, full_rank_grad)

    def _natural_gradient_transform(self, low_rank_grad):
        """
        Transforms the low-rank gradients using the natural gradient.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        # Flatten the low-rank gradient tensor to a vector
        grad_vector = low_rank_grad.reshape(-1)  # Shape: [k]

        # Ensure the gradient is on the same device as the history
        device = grad_vector.device

        # Add the current gradient to the history buffer
        if len(self.grad_history) >= self.history_size:
            self.grad_history.pop(0)  # Remove the oldest gradient
        self.grad_history.append(grad_vector.detach())

        # Stack gradients to form G ∈ ℝ^{k x s}, where s is the history size
        G = torch.stack(self.grad_history, dim=1)  # Shape: [k, s]
        G = G.to(device)  # Ensure G is on the correct device

        lambda_inv = 1.0 / self.lambda_damping

        # Compute G^T * G ∈ ℝ^{s x s}
        GTG = torch.mm(G.t(), G)  # Shape: [s, s]

        # Compute S = I + λ^{-1} * G^T * G
        S = lambda_inv * GTG
        S.diagonal().add_(1.0)  # In-place addition to the diagonal

        # Use Cholesky decomposition to solve S_inv * (G^T * grad_vector)
        # Compute G^T * grad_vector ∈ ℝ^{s}
        GTg = torch.mv(G.t(), grad_vector)  # Shape: [s]

        # Perform Cholesky decomposition
        L = torch.linalg.cholesky(S)  # S = L @ L.T, L is lower triangular

        # Solve for temp in S * temp = G^T * grad_vector
        temp = torch.cholesky_solve(GTg.unsqueeze(1), L)  # Shape: [s, 1]
        temp = temp.squeeze(1)  # Shape: [s]

        # Compute G @ temp ∈ ℝ^{k}
        G_temp = torch.mv(G, temp)  # Shape: [k]

        # Compute natural gradient: ng = λ^{-1} * grad_vector - λ^{-2} * G_temp
        ng_vector = (lambda_inv * grad_vector) - ((lambda_inv**2) * G_temp)

        # Reshape back to the original shape of low_rank_grad
        natural_grad = ng_vector.view_as(low_rank_grad)

        return natural_grad

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.factors, low_rank_grad)
        if self.proj_type in ["online_galore", "online_natural_galore"]:
            # Update factors after projection back
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
        return full_rank_grad * self.scale

    def _update_factors(self, full_rank_grad, lr_ratio):
        """
        Updates the factors using gradient descent.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            lr_ratio (float): Ratio of current learning rate to initial learning rate.
        """
        with torch.enable_grad():
            # Set all factors to require gradients
            for factor in self.factors:
                factor.requires_grad = True

            # Zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            # Compute loss for updating factors
            normalized_full_rank_grad = full_rank_grad / torch.norm(full_rank_grad)
            approx_grad = self.inverse_transform(
                self.factors, self.transform(self.factors, normalized_full_rank_grad)
            )
            loss = torch.norm(normalized_full_rank_grad - approx_grad) ** 2

            # Backpropagate
            loss.backward()

            # Update learning rates based on lr_ratio
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = (1.0 / self.update_proj_gap) * lr_ratio

            for factor in self.factors:
                factor.requires_grad = False
            # Note: Optimizer steps are taken in project_back after projection

    def get_orthogonal_matrix(self, weights, rank):
        """
        Computes the Tucker decomposition of the weights.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank (int or tuple): The desired rank for each mode.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The core tensor and factors.
        """
        # Ensure weights are float32
        original_dtype = weights.dtype
        if weights.dtype != torch.float32:
            weights = weights.float()

        # Determine the ranks for each mode
        if isinstance(rank, int):
            # For higher-order tensors, apply the same rank to all modes
            ranks = [rank] * weights.ndim
        elif isinstance(rank, (list, tuple)):
            # Ensure that the length of rank matches the number of dimensions
            if len(rank) != weights.ndim:
                raise ValueError(
                    f"Rank tuple length {len(rank)} does not match tensor dimensions {weights.ndim}."
                )
            ranks = rank
        else:
            raise ValueError("Rank must be an integer or a tuple/list of integers.")

        # Perform Tucker decomposition
        core, factors = tucker(weights, rank=ranks)

        # Convert factors back to original dtype if necessary
        factors = [factor.to(dtype=original_dtype) for factor in factors]

        return core, factors

    def transform(self, factors, x):
        """
        Transforms the input tensor using the factors.

        Args:
            factors (List[torch.Tensor]): Factors from the Tucker decomposition.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        """
        Reconstructs the tensor from the low-rank representation.

        Args:
            factors (List[torch.Tensor]): Factors from the Tucker decomposition.
            x (torch.Tensor): The low-rank representation.

        Returns:
            torch.Tensor: The reconstructed tensor.
        """
        return tenalg.multi_mode_dot(x, factors)
