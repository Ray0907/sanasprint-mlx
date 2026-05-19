from __future__ import annotations


def build_vae_decoder_mapping_report(tensor_infos: list[dict]) -> dict:
    decoder_keys = []
    unresolved = []
    for info in tensor_infos:
        if info.get("component") != "vae":
            continue
        key = info.get("key", "")
        if key.startswith("decoder."):
            decoder_keys.append(key)
        else:
            unresolved.append(key)
    if unresolved:
        raise ValueError(f"unknown VAE entries require review: {', '.join(unresolved)}")
    return {
        "decoder_keys": sorted(decoder_keys),
        "decoder_key_count": len(decoder_keys),
    }
