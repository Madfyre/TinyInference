# Mixture-of-Experts Top-K Router (FP16)

Mixtral-style expert selection: per-token Top-K over FP16 router logits.

**Techniques:** Top-K selection with stable lowest-index tie-breaking; emits selected expert indices and their logits.
