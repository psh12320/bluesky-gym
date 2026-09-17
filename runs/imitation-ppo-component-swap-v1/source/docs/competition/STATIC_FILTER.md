# Static-area support for learned traffic avoidance

The optional --static-filter training flag selects the existing StaticActionProjection for restricted areas and sector boundaries. It leaves aircraft-to-aircraft conflict filtering off. The existing --filter flag continues to select both static and traffic filtering, including when both flags are set. Neither flag changes its underlying controller.

Route guidance is independent. With guidance enabled, the local actor observes the existing reference route and learns heading offsets and speed changes. Static projection may adjust a heading for a restricted area or sector boundary; it does not consult other aircraft or modify speed. The existing approximate predictor does not guarantee safety.

Training checkpoints record the requested support. Evaluation reconstructs it from the checkpoint configuration and records the actual simulator mixins. Missing static_filter fields in older checkpoints default to false. Comparisons and learning curves reject mismatched support settings.

The fixed classical benchmark retains guidance plus both filters. Existing running experiments use their own immutable source bundles. This option belongs to a separate experiment and is rejected by the registered legacy support matrix.

A future performance experiment must evaluate each trained policy against its own untrained initial policy with identical support, use multiple training seeds, retain native physical metrics, and report static-area and traffic violations separately. If comparing closest-approach features with zeroed feature channels, keep network capacity and initial tensors identical. Smoke tests only verify integration; they establish no learning benefit. The reserved unseen streams remain unused until candidate selection.
