"""Hydra configuration schema for the RFDiffusion input args.

See config/inference/base.yaml for the matching configurations.
"""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


def register_configs() -> None:
    """Registers the base configuration to the ConfigStore instance."""
    cs = ConfigStore.instance()
    cs.store(name="base", node=BaseConfig)


@dataclass
class InferenceConfig:  # noqa: D101
    input_pdb: Optional[str] = None
    num_designs: int = 10
    design_startnum: int = 0
    ckpt_override_path: Optional[str] = None
    symmetry: Optional[str] = None
    recenter: bool = True
    radius: float = 10.0
    model_only_neighbors: bool = False
    output_prefix: str = "samples/design"
    write_trajectory: bool = True
    scaffold_guided: bool = False
    model_runner: str = "SelfConditioning"
    cautious: bool = True
    align_motif: bool = True
    symmetric_self_cond: bool = True
    final_step: int = 1
    deterministic: bool = False
    trb_save_ckpt_path: Optional[str] = None
    schedule_directory_path: Optional[str] = None
    model_directory_path: Optional[str] = None


@dataclass
class ContigMapConfig:  # noqa: D101
    contigs: Optional[list[str]] = None
    inpaint_seq: Optional[list[str]] = None
    inpaint_str: Optional[str] = None
    inpaint_str_helix: Optional[str] = None
    inpaint_str_strand: Optional[str] = None
    inpaint_str_loop: Optional[str] = None
    provide_seq: Optional[list[str]] = None
    length: Optional[int] = None


@dataclass
class SE3ParamFull:  # noqa: D101
    num_layers: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    n_heads: int = 4
    div: int = 4
    l0_in_features: int = 8
    l0_out_features: int = 8
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 32


@dataclass
class SE3ParamTopK:  # noqa: D101
    num_layers: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    n_heads: int = 4
    div: int = 4
    l0_in_features: int = 64
    l0_out_features: int = 64
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 64


@dataclass
class ModelConfig:  # noqa: D101
    n_extra_block: int = 4
    n_main_block: int = 32
    n_ref_block: int = 4
    d_msa: int = 256
    d_msa_full: int = 64
    d_pair: int = 128
    d_templ: int = 64
    n_head_msa: int = 8
    n_head_pair: int = 4
    n_head_templ: int = 4
    d_hidden: int = 32
    d_hidden_templ: int = 32
    p_drop: float = 0.15
    SE3_param_full: SE3ParamFull = field(default_factory=SE3ParamFull)
    SE3_param_topk: SE3ParamTopK = field(default_factory=SE3ParamTopK)
    freeze_track_motif: bool = False
    use_motif_timestep: bool = False


@dataclass
class DiffuserConfig:  # noqa: D101
    T: int = 50
    b_0: float = 1e-2
    b_T: float = 7e-2
    schedule_type: str = "linear"
    so3_type: str = "igso3"
    crd_scale: float = 0.25
    partial_T: Optional[int] = None
    so3_schedule_type: str = "linear"
    min_b: float = 1.5
    max_b: float = 2.5
    min_sigma: float = 0.02
    max_sigma: float = 1.5


@dataclass
class DenoiserConfig:  # noqa: D101
    noise_scale_ca: float = 1
    final_noise_scale_ca: float = 1
    ca_noise_schedule_type: str = "constant"
    noise_scale_frame: float = 1
    final_noise_scale_frame: float = 1
    frame_noise_schedule_type: str = "constant"


@dataclass
class PpiConfig:  # noqa: D101
    hotspot_res: Optional[list[str]] = None


@dataclass
class PotentialsConfig:  # noqa: D101
    guiding_potentials: Optional[str] = None
    guide_scale: int = 10
    guide_decay: str = "constant"
    olig_inter_all: Optional[str] = None
    olig_intra_all: Optional[str] = None
    olig_custom_contact: Optional[str] = None
    substrate: Optional[str] = None


@dataclass
class ContigSettingsConfig:  # noqa: D101
    ref_idx: Optional[str] = None
    hal_idx: Optional[str] = None
    idx_rf: Optional[str] = None
    inpaint_seq_tensor: Optional[str] = None


@dataclass
class PreprocessConfig:  # noqa: D101
    sidechain_input: bool = False
    motif_sidechain_input: bool = True
    d_t1d: int = 22
    d_t2d: int = 44
    prob_self_cond: float = 0.0
    str_self_cond: bool = False
    predict_previous: bool = False


@dataclass
class LoggingConfig:  # noqa: D101
    inputs: bool = False


@dataclass
class ScaffoldGuidedConfig:  # noqa: D101
    scaffoldguided: bool = False
    target_pdb: bool = False
    target_path: Optional[str] = None
    scaffold_list: Optional[str] = None
    scaffold_dir: Optional[str] = None
    sampled_insertion: int = 0
    sampled_N: int = 0
    sampled_C: int = 0
    ss_mask: int = 0
    systematic: bool = False
    target_ss: Optional[str] = None
    target_adj: Optional[str] = None
    mask_loops: bool = True
    contig_crop: Optional[str] = None


@dataclass
class DuoStateConfig:
    """Configs for multi-state design.

    Attributes:
        pdb_path: path to the second PDB file.
        contigmap: ContigMapConfig for the second PDB file. Should be identical for the
            parts to be generated.
        ppi: hotspot residues for the second PDB file.
    """

    pdb_path: Optional[str] = None
    contigmap: ContigMapConfig = field(default_factory=ContigMapConfig)
    ppi: PpiConfig = field(default_factory=PpiConfig)


@dataclass
class BaseConfig(DictConfig):  # noqa: D101
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    contigmap: ContigMapConfig = field(default_factory=ContigMapConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffuser: DiffuserConfig = field(default_factory=DiffuserConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    ppi: PpiConfig = field(default_factory=PpiConfig)
    potentials: PotentialsConfig = field(default_factory=PotentialsConfig)
    contig_settings: ContigSettingsConfig = field(default_factory=ContigSettingsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    scaffoldguided: ScaffoldGuidedConfig = field(default_factory=ScaffoldGuidedConfig)
    duostate: DuoStateConfig = field(default_factory=DuoStateConfig)
