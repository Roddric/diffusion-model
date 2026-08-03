"""Phase 2D benchmark with validation path-energy checkpoint selection."""

from phase2c_benchmark import run_benchmark


def main():
    run_benchmark(
        phase="2D",
        diffusion_name="Phase2D-PathSelected-Diffusion",
        selection_metric="sampled_path_energy",
        output_default="../prod_output/phase2d_benchmark.json",
        checkpoint_default=(
            "../prod_output/phase2d_diffusion_seed{seed}.pt"
        ),
        previous_filename="phase2c_benchmark.json",
        previous_model_name="Phase2C-SplitHead-Diffusion",
        previous_label="phase2c",
    )


if __name__ == "__main__":
    main()
