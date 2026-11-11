"""Re-exports the submitted LegalLM implementation (phase1_pretraining/legallm/model.py).

The file is loaded by path, not copied, so the backbone used by every phase is the exact code that was
trained and shipped with the checkpoint.
"""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "phase1_pretraining", "legallm", "model.py")
_spec = importlib.util.spec_from_file_location("legallm_model", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LegalLM = _mod.LegalLM
ModelConfig = _mod.ModelConfig
RMSNorm = _mod.RMSNorm
Attention = _mod.Attention
SwiGLU = _mod.SwiGLU
Block = _mod.Block
rope_tables = _mod.rope_tables
apply_rope = _mod.apply_rope
