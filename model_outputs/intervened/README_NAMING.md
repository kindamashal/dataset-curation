# Intervention Results Naming Guide

All files in this directory follow a standardized naming convention to make the experimental setup clear at a glance.

## Naming Template
`[Target]_[Modality]_[Action]_[Modulator]_[Alpha]_[Features]_[Constraint].json`

### 1. Target (Question Concept)
Which concept questions were asked:
- `red`, `green`, `blue`: Primary color concepts.
- `gen`: General knowledge (irrelevant concept/control).

### 2. Modality
(Optional) Specific modality of the target questions:
- `img`: Image-based color assessment.
- `txt`: Text-based color assessment (if distinguishing from images).

### 3. Action (Intervention Type)
What was done to the features:
- `base`: No intervention (Baseline).
- `ablate`: Features set to 0.
- `invert`: Features multiplied by a negative alpha (-100, -1000).
- `steer`: Features amplified/scaled (positive alpha).

### 4. Modulator (Steering Concept)
(Optional) Which color features were used to intervene. If omitted, the modulator is the same as the target concept.
- `red`, `green`, `blue`: Primary color SAE features.
- `red_new`, `green_new`, `blue_txt`: Specific variations or source features.

### 5. Alpha (Strength)
The magnitude of the intervention:
- `a0`: Alpha = 0 (Ablation).
- `a100`, `a500`, `a1000`, `a10000`, `a100000`: Scaling factors.

### 6. Features (Feature Set)
Which specific SAE features were used:
- `std`: Standard set.
- `foi`: Features of Interest (curated set).
- `excl`: Exclusive features (no overlap with other colors).
- `excl_rb`: Exclusive features specifically filtered for Red-Blue overlap.
- `fused`, `new`: Experimental or merged feature sets.

### 7. Constraint (Prompt/Output)
(Optional) Specific prompt or generation constraints:
- `unconstr`: Unconstrained (multianswer) generation.
- `strict`: Strict prompt (no empty strings allowed).

---

### Examples:
- `red_steer_blue_a100_foi.json`: Red questions, steered with Blue features, Alpha 100, using FOI set.
- `blue_img_invert_a1000_std.json`: Blue image questions, Blue features inverted, Alpha 1000, standard set.
- `red_img_base.json`: Red image questions baseline.
- `green_steer_red_new_a10000_std.json`: Green questions, steered with "red_new" features, Alpha 10,000.
