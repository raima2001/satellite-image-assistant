---
name: ndvi-band-guidance
description: Explains how to compute a vegetation index like NDVI from Sentinel-2 band names, and which asset names correspond to red and near-infrared. Load when the analyst asks about NDVI, vegetation index, or which bands to use for vegetation.
---

# NDVI band guidance

Sentinel-2 L2A assets typically include `red` (B04) and `nir` (B08). NDVI is
computed as (NIR - Red) / (NIR + Red), using the surface reflectance values of
those two bands.

When asked which bands to use for a vegetation index: name the asset keys the
scene's `get_scene_details` call actually returned (e.g. `red`, `nir`) -
never claim a band is available if it is not literally in that list. If the
scene's assets do not include both a red and a near-infrared equivalent, say
so explicitly.

Computing NDVI itself happens outside this tool call - you are telling the
analyst which bands to use and the formula, not calculating pixel values.

Do not add wavelengths, spatial resolutions, or other metadata absent
from the tool output. The band mappings and NDVI formula in this skill
are general guidance, not metadata retrieved for this scene.