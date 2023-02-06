from .deeplab import DeeplabV3
from .residual import IdentityResidualBlock, ResidualBlock
from .misc import GlobalAvgPool2d
from .segformer import SegFormer, segformer_type_args, SegFormer_Head, SegFormer_Body
from .DSSNet import DSS, build_model