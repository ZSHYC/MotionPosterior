# MotionPosterior motion contract

`MotionPosteriorNet` is the final motion-aware model. `TrackNetV2` and `TrackNetV5`
remain as historical baselines. The current dataset is tennis-ball tracking, while
the contract itself describes generic tiny moving point targets.

This repository supports three distinct temporal protocols:

* **3-frame center**: `[t-1, t, t+1] -> y_t` for offline inference.
* **5-frame center**: `[t-2, t-1, t, t+1, t+2] -> y_t` for offline high-accuracy inference.
* **5-frame causal**: `[t-4, t-3, t-2, t-1, t] -> y_t` for online inference.

They must be reported as separate experiments. A five-frame model that emits
five frame predictions is a fourth protocol and must not be compared directly
with a center-only model.

## Output and supervision

`MotionPosteriorNet(return_aux=True)` emits a heatmap plus normalized center,
visibility, uncertainty, velocity, and acceleration heads. The upgraded loss
uses batch `coords` and `visibility` when available. Coordinates can be pixel
coordinates or normalized coordinates; pixel values are normalized using the
model target dimensions. If metadata is absent, the rendered heatmap mass is
used for backwards compatibility.

Position, uncertainty, velocity, and acceleration losses are masked whenever a
frame is invisible. Velocity and acceleration use `dt` when the dataloader
provides it, or `1 / fps`, and otherwise use unit frame intervals.

The deployment decoder should consume the rich output contract in this order:

```text
heatmap candidate -> offset refinement -> visibility gate -> uncertainty
```

The fixed threshold/largest-contour decoder remains a legacy baseline only.

## Motion architecture direction

The intended next Motion iteration is camera-compensated local matching:

```text
RGB/high-resolution features + temporal difference features
    -> camera transform candidate and confidence
    -> velocity-conditioned multi-scale local correlation
    -> top-k candidate refinement
    -> position / visibility / uncertainty / kinematics
```

The camera path must be confidence-gated, since raw optical flow can confuse
camera movement, player/racket motion, and a blurred ball. The local search
radius should depend on predicted velocity and uncertainty instead of a single
fixed radius. Any future acceleration regularizer must be robust around hits,
bounces, and other change points.
