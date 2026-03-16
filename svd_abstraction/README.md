# SVD Abstraction

This folder is a clean workspace for the `base -> abs` residual-abstraction
experiments.

Current baseline choice:

- Start from the lighter `hierarchy/gbp/gbp.py` style solver because its core
  GBP data model is much easier to reason about.
- Add only the minimum residual-facing hooks we need for the next step:
  variable residual evaluation and residual-priority iteration.
- Do not copy the full `raylib_gbp` multigrid stack yet. It mixes together AMG
  hierarchy construction, visualization, activation logic, and coarse/fine
  bookkeeping that we do not want to inherit wholesale.

Planned direction:

- Keep grouping metadata from the abstraction layer.
- Build one local SVD basis `B_g` per group.
- Freeze `B_g` after an initial base warm-up.
- Run residual correction directly between `base` and `abs`, instead of running
  GBP on an intermediate `super` state layer.

What is implemented now:

- `pose_graph.py`: a small linear pose-graph problem builder.
- `grouping.py`: order / grid / kmeans grouping helpers.
- `residual_abstraction.py`: frozen-basis residual restriction, coarse solve,
  and prolongated correction back to the base graph.
- `demo_pose_graph.py`: a minimal end-to-end example.

You can run the demo with either:

- `python -m svd_abstraction.demo_pose_graph`
- `python svd_abstraction/demo_pose_graph.py`

So the intent here is not "copy one old file verbatim", but "use the hierarchy
solver as the clean base and port over only the residual-correction semantics
that matter from raylib".
