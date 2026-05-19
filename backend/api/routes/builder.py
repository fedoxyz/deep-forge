"""Routes for the visual model builder: catalog, presets, validation, build, code export."""

from fastapi import APIRouter, HTTPException

from backend.api.models import ModelBuildRequest, ModelValidateRequest

router = APIRouter(prefix="/api/builder", tags=["builder"])


@router.get("/catalog")
async def get_layer_catalog():
    from backend.core.model_builder import get_layer_catalog
    return {"layers": get_layer_catalog()}


@router.get("/presets")
async def get_model_presets():
    from backend.core.model_builder import MODEL_PRESETS
    return {"presets": MODEL_PRESETS}


@router.post("/validate")
async def validate_model(req: ModelValidateRequest):
    from backend.core.model_builder import validate_model_spec, infer_shapes

    errors = validate_model_spec(req.spec)
    shapes = None
    if not errors and req.input_shape:
        shapes = infer_shapes(req.spec, tuple(req.input_shape))
    return {"errors": errors, "layers_with_shapes": shapes}


@router.post("/build")
async def build_model(req: ModelBuildRequest):
    from backend.core.model_builder import build_model, validate_model_spec

    errors = validate_model_spec(req.spec)
    if errors:
        raise HTTPException(400, {"errors": errors})

    model = build_model(req.spec)
    total = sum(p.numel() for p in model.parameters())
    return {
        "status": "ok",
        "model_name": req.spec.get("name", "Custom"),
        "total_params": total,
        "total_params_human": f"{total / 1e6:.2f}M" if total > 1e6 else f"{total / 1e3:.1f}K",
        "num_layers": len(req.spec.get("layers", [])),
    }


@router.post("/to_code")
async def model_to_code(req: ModelBuildRequest):
    from backend.core.model_builder import model_spec_to_code
    return {"code": model_spec_to_code(req.spec)}

@router.get("/expand_macro")
async def expand_macro_endpoint(type: str, dim: int = 256, heads: int = 8,
                                 mlp_ratio: float = 4.0, drop_path: float = 0.0):
    """Returns the primitive layer list that a macro block expands into."""
    from backend.core.model_builder import MACRO_EXPANSIONS
    if type not in MACRO_EXPANSIONS:
        raise HTTPException(404, f"No macro for '{type}'")
    params = {'dim': dim, 'heads': heads, 'mlp_ratio': mlp_ratio, 'drop_path': drop_path}
    return {'layers': MACRO_EXPANSIONS[type](params)}

@router.get("/block_info")
async def block_info(type: str, dim: int = 256, heads: int = 8, mlp_ratio: float = 4.0):
    """Approximate parameter count for a single block."""
    counts = {
        'rope_encoder_block': lambda d, h, r: int(4*d*d + 2*d*d*r),  # attn + ffn
        'decoder_block':      lambda d, h, r: int(6*d*d + 2*d*d*r),
        'patch_embed':        lambda d, h, r: 3*16*16*d,
        'swiglu_ffn':         lambda d, h, r: int(3*d*d*r),
        'sdpa_attention':     lambda d, h, r: 4*d*d,
        'cross_attention':    lambda d, h, r: 4*d*d,
    }
    fn = counts.get(type)
    if not fn:
        return {'approx_params': 0}
    return {'approx_params': fn(dim, heads, mlp_ratio)}
