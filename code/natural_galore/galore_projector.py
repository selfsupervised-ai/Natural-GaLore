import torch


class GaLoreProjector:
    def __init__(self, rank, update_proj_gap=200, scale=1.0, proj_type="std"):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix = None
        self.ortho_matrix_optim = None

    def project(self, full_rank_grad, step, lr_ratio=1.0):
        if self.proj_type == "std":
            low_rank_grad = self._std_projection(full_rank_grad, step)
        elif "continuous" in self.proj_type:
            low_rank_grad = self._continuous_projection(full_rank_grad, step, lr_ratio)
        else:
            raise NotImplementedError(
                f"Projection type '{self.proj_type}' is not implemented."
            )
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == "std":
            full_rank_grad = self._std_project_back(low_rank_grad)
        elif "continuous" in self.proj_type:
            full_rank_grad = self._continuous_project_back(low_rank_grad)
        else:
            raise NotImplementedError(
                f"Projection type '{self.proj_type}' is not implemented."
            )
        return full_rank_grad * self.scale

    def _std_projection(self, full_rank_grad, step):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode="right"
                )
            return torch.matmul(full_rank_grad, self.ortho_matrix.t())
        else:
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode="left"
                )
            return torch.matmul(self.ortho_matrix.t(), full_rank_grad)

    def _std_project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            return torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            return torch.matmul(self.ortho_matrix, low_rank_grad)

    def _continuous_projection(self, full_rank_grad, step, lr_ratio):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode="right"
                )
                self.ortho_matrix.requires_grad = True
                self.ortho_matrix_optim = torch.optim.AdamW(
                    [self.ortho_matrix], lr=1 / self.update_proj_gap
                )
            else:
                if step % self.update_proj_gap == 0:
                    self._update_ortho_matrix(full_rank_grad, lr_ratio)
            return torch.matmul(full_rank_grad, self.ortho_matrix.t())
        else:
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode="left"
                )
                self.ortho_matrix.requires_grad = True
                self.ortho_matrix_optim = torch.optim.AdamW(
                    [self.ortho_matrix], lr=1 / self.update_proj_gap
                )
            else:
                if step % self.update_proj_gap == 0:
                    self._update_ortho_matrix(full_rank_grad, lr_ratio)
            return torch.matmul(self.ortho_matrix.t(), full_rank_grad)

    def _continuous_project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        # Update the ortho_matrix optimizer
        self.ortho_matrix_optim.step()
        self.ortho_matrix.grad = None
        return full_rank_grad

    def _update_ortho_matrix(self, full_rank_grad, lr_ratio):
        self.ortho_matrix.grad = None
        projection = self.ortho_matrix @ self.ortho_matrix.t()
        normalized_grad = full_rank_grad / full_rank_grad.norm()
        loss = (normalized_grad - projection @ normalized_grad).norm() ** 2
        loss.backward()
        # Update optimizer learning rate based on lr_ratio
        for group in self.ortho_matrix_optim.param_groups:
            group["lr"] = (1 / self.update_proj_gap) * lr_ratio

    def get_orthogonal_matrix(self, weights, rank, mode):
        # Save original data type and device
        original_dtype = weights.dtype
        original_device = weights.device

        # Convert weights to float32 if necessary
        if weights.dtype != torch.float32:
            weights_float = weights.float()
        else:
            weights_float = weights

        # Perform SVD on float32 weights
        U, s, Vh = torch.linalg.svd(weights_float, full_matrices=False)

        # Extract the orthogonal matrix based on mode
        if mode == "right":
            ortho_matrix = Vh[:rank, :].detach()
        elif mode == "left":
            ortho_matrix = U[:, :rank].detach()
        else:
            raise ValueError("Mode should be 'left' or 'right'.")

        # Convert orthogonal matrix back to original dtype and device if necessary
        if (
            ortho_matrix.dtype != original_dtype
            or ortho_matrix.device != original_device
        ):
            ortho_matrix = ortho_matrix.to(dtype=original_dtype, device=original_device)

        return ortho_matrix
