import torch
from beartype import beartype
from diffdrr.pose import convert
from jaxtyping import Float, jaxtyped
from diffdrr.pose import RigidTransform


class GeodesicSE3(torch.nn.Module):
    """Calculate the distance between transforms in the log-space of SE(3)."""

    def __init__(self):
        super().__init__()

    @beartype
    @jaxtyped
    def forward(
        self,
        pose_1: RigidTransform,
        pose_2: RigidTransform,
    ) -> Float[torch.Tensor, "b"]:
        return pose_2.compose(pose_1.inverse()).get_se3_log().norm(dim=1)
