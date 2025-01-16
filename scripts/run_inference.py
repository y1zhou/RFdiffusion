#!/usr/bin/env python
"""Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import glob
import logging
import os
import pickle
import random
import re
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from rfdiffusion.config_schema import BaseConfig, register_configs
from rfdiffusion.inference import utils as iu
from rfdiffusion.util import writepdb, writepdb_multi

register_configs()


def make_deterministic(seed=0):  # noqa: D103
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_sampled_results(
    denoised_xyz_stack: list[torch.Tensor],
    px0_xyz_stack: list[torch.Tensor],
    seq_stack: list[torch.Tensor],  # TODO: save all seqs instead of just the final one
    plddt_stack: list[torch.Tensor],
    seq_init: torch.Tensor,
    binder_len: int,
    chain_ids: list[str],
    out_prefix: str,
    out_suffix: str = "",
    write_trajectory: bool = True,
) -> torch.Tensor:
    """Save sampled results to pdb files.

    Args:
        denoised_xyz_stack: List of denoised coordinates (what went into the model at each timestep).
        px0_xyz_stack: List of pX0 coordinates (what the model predicted at each timestep).
        seq_stack: List of sequences.
        plddt_stack: List of pLDDT values.
        seq_init: Initial sequence.
        binder_len: Length of the binder.
        chain_ids: Chain IDs.
        out_prefix: Output prefix.
        out_suffix: Output suffix.
        write_trajectory: Whether to output full trajectories.
    """
    # Flip order for better visualization in pymol
    denoised_xyz = torch.stack(denoised_xyz_stack)
    denoised_xyz = torch.flip(denoised_xyz, [0])
    px0_xyz = torch.stack(px0_xyz_stack)
    px0_xyz = torch.flip(px0_xyz, [0])

    # For logging -- don't flip
    plddt = torch.stack(plddt_stack)

    # Save outputs
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    final_seq = seq_stack[-1]

    # Output glycines, except for motif region
    final_seq = torch.where(
        torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
    )  # 7 is glycine

    bfacts = torch.ones_like(final_seq.squeeze())
    # make bfact=0 for diffused coordinates
    bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0

    # pX0 last step
    # Now don't output sidechains
    writepdb(
        f"{out_prefix}{out_suffix}.pdb",
        denoised_xyz[0, :, :4],
        final_seq,
        binder_len,
        chain_idx=chain_ids,
        bfacts=bfacts,
    )

    if write_trajectory:
        # trajectory pdbs
        traj_prefix = (
            os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
        )
        os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

        writepdb_multi(
            f"{traj_prefix}{out_suffix}_Xt-1_traj.pdb",
            denoised_xyz,
            bfacts,
            final_seq.squeeze(),
            use_hydrogens=False,
            backbone_only=False,
            chain_ids=chain_ids,
        )
        writepdb_multi(
            f"{traj_prefix}{out_suffix}_pX0_traj.pdb",
            px0_xyz,
            bfacts,
            final_seq.squeeze(),
            use_hydrogens=False,
            backbone_only=False,
            chain_ids=chain_ids,
        )

    return plddt


@hydra.main(version_base=None, config_path="../config/inference", config_name="base")
def main(conf: BaseConfig) -> None:  # noqa: D103
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        device_name = torch.cuda.get_device_name(device)
        log.info(
            f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}"
        )
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")
        device = torch.device("cpu")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(r".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue

        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []
        denoised_xyz_stack2 = []
        px0_xyz_stack2 = []
        seq_stack2 = []
        plddt_stack2 = []

        if sampler.inf_conf.model_runner == "DuoStateSampler":
            x_init, x_init2 = x_init
            seq_init, seq_init2 = seq_init
            x_t2 = torch.clone(x_init2).to(device)
            seq_t2 = torch.clone(seq_init2).to(device)
        x_t = torch.clone(x_init).to(device)
        seq_t = torch.clone(seq_init).to(device)
        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            if sampler.inf_conf.model_runner == "DuoStateSampler":
                (
                    px0_s1,
                    x_t,
                    seq_t,
                    plddt_s1,
                    px0_s2,
                    x_t_1_s2,
                    seq_t_1_s2,
                    plddt_s2,
                ) = sampler.sample_step(
                    t=t,
                    x_t=x_t,
                    seq_init=seq_t,
                    final_step=sampler.inf_conf.final_step,
                    x_t2=x_t2,
                    seq_init2=seq_t2,
                )
                px0_xyz_stack.append(px0_s1)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                plddt_stack.append(plddt_s1[0])  # remove singleton leading dimension
                px0_xyz_stack2.append(px0_s2)
                denoised_xyz_stack2.append(x_t_1_s2)
                seq_stack2.append(seq_t_1_s2)
                plddt_stack2.append(plddt_s2[0])
            else:
                px0, x_t, seq_t, plddt = sampler.sample_step(
                    t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
                )
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                plddt_stack.append(plddt[0])  # remove singleton leading dimension

        if sampler.inf_conf.model_runner == "DuoStateSampler":
            plddt_s1 = save_sampled_results(
                denoised_xyz_stack,
                px0_xyz_stack,
                seq_stack,
                plddt_stack,
                seq_init,
                sampler.binderlen,
                sampler.chain_idx,
                out_prefix,
                out_suffix="_s1",
                write_trajectory=sampler.inf_conf.write_trajectory,
            )
            plddt_s2 = save_sampled_results(
                denoised_xyz_stack2,
                px0_xyz_stack2,
                seq_stack2,
                plddt_stack2,
                seq_init2,
                sampler.binderlen,
                sampler.chain_idx2,
                out_prefix,
                out_suffix="_s2",
                write_trajectory=sampler.inf_conf.write_trajectory,
            )
            trb = dict(
                config=OmegaConf.to_container(sampler._conf, resolve=True),
                plddt=plddt_s1.detach().cpu().numpy(),
                duostate_plddt=plddt_s2.detach().cpu().numpy(),
                device=torch.cuda.get_device_name(device)
                if torch.cuda.is_available()
                else "CPU",
                time=time.time() - start_time,
            )
        else:
            plddt = save_sampled_results(
                denoised_xyz_stack,
                px0_xyz_stack,
                seq_stack,
                plddt_stack,
                seq_init,
                sampler.binderlen,
                sampler.chain_idx,
                out_prefix,
                write_trajectory=sampler.inf_conf.write_trajectory,
            )
            trb = dict(
                config=OmegaConf.to_container(sampler._conf, resolve=True),
                plddt=plddt.detach().cpu().numpy(),
                device=torch.cuda.get_device_name(device)
                if torch.cuda.is_available()
                else "CPU",
                time=time.time() - start_time,
            )

        # run metadata
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        if hasattr(sampler, "contig_map2"):
            for key, value in sampler.contig_map2.get_mappings().items():
                trb[f"duostate_{key}"] = value
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        log.info(f"Finished design in {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
