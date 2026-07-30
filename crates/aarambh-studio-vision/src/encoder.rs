use std::collections::HashMap;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias, ops::softmax,
};
use serde::{Deserialize, Serialize};

/// CLIP-style ViT encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionEncoderConfig {
    /// Patch side length in pixels.
    pub patch_size: usize,
    /// Square input image side length in pixels.
    pub image_size: usize,
    /// Number of RGB input channels.
    pub in_channels: usize,
    /// Vision transformer hidden width.
    pub vit_d_model: usize,
    /// Number of transformer encoder blocks.
    pub vit_layers: usize,
    /// Number of attention heads.
    pub vit_heads: usize,
    /// MLP hidden width inside each transformer block.
    pub mlp_dim: usize,
    /// Number of non-CLS patch tokens.
    pub num_patches: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            patch_size: 32,
            image_size: 224,
            in_channels: 3,
            vit_d_model: 768,
            vit_layers: 12,
            vit_heads: 12,
            mlp_dim: 3072,
            num_patches: 49,
            norm_eps: 1e-5,
        }
    }
}

impl VisionEncoderConfig {
    /// Load a vision encoder configuration from JSON.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config = serde_json::from_reader(file)?;
        Ok(config)
    }

    /// Validate encoder dimensions.
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0
            || self.image_size == 0
            || self.in_channels == 0
            || self.vit_d_model == 0
            || self.vit_layers == 0
            || self.vit_heads == 0
            || self.mlp_dim == 0
        {
            return Err(AarambhError::Config(
                "vision encoder dimensions must be non-zero".into(),
            ));
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(AarambhError::Config(format!(
                "image_size {} must be divisible by patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        let patches_per_side = self.image_size / self.patch_size;
        let expected_patches = patches_per_side * patches_per_side;
        if self.num_patches != expected_patches {
            return Err(AarambhError::Config(format!(
                "num_patches {} does not match image_size/patch_size grid {}",
                self.num_patches, expected_patches
            )));
        }
        if !self.vit_d_model.is_multiple_of(self.vit_heads) {
            return Err(AarambhError::Config(
                "vit_d_model must be divisible by vit_heads".into(),
            ));
        }
        Ok(())
    }

    /// Return flattened patch width.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch_size * self.patch_size
    }

    /// Return per-head attention width.
    pub fn head_dim(&self) -> usize {
        self.vit_d_model / self.vit_heads
    }
}

/// Frozen CLIP-style vision transformer encoder.
#[derive(Debug, Clone)]
pub struct ClipVisionEncoder {
    config: VisionEncoderConfig,
    patch_embed: Linear,
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_norm: LayerNorm,
    blocks: Vec<VisionBlock>,
    post_norm: LayerNorm,
}

impl ClipVisionEncoder {
    /// Build the encoder from a variable builder.
    pub fn new(config: VisionEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let patch_embed =
            linear_no_bias(config.patch_dim(), config.vit_d_model, vb.pp("patch_embed"))?;
        let class_embedding = vb.get_with_hints(
            config.vit_d_model,
            "class_embedding",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let position_embedding = vb.get_with_hints(
            (config.num_patches + 1, config.vit_d_model),
            "position_embedding",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;
        let pre_norm = layer_norm(config.vit_d_model, config.norm_eps, vb.pp("pre_norm"))?;
        let post_norm = layer_norm(config.vit_d_model, config.norm_eps, vb.pp("post_norm"))?;
        let mut blocks = Vec::with_capacity(config.vit_layers);
        for idx in 0..config.vit_layers {
            blocks.push(VisionBlock::new(&config, vb.pp("blocks").pp(idx))?);
        }
        Ok(Self {
            config,
            patch_embed,
            class_embedding,
            position_embedding,
            pre_norm,
            blocks,
            post_norm,
        })
    }

    /// Load pretrained encoder weights from SafeTensors.
    pub fn load_pretrained(
        path: impl AsRef<Path>,
        config: VisionEncoderConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path.as_ref(), device)?;
        let tensors = normalize_clip_tensors(tensors, &config)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        Self::new(config, vb)
    }

    /// Return the encoder configuration.
    pub fn config(&self) -> &VisionEncoderConfig {
        &self.config
    }

    /// Encode `[batch, 3, image_size, image_size]` pixels into non-CLS patch tokens.
    pub fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let patches = self.patchify(images)?;
        let batch = patches.dims()[0];
        let mut x = self.patch_embed.forward(&patches)?;
        let cls = self
            .class_embedding
            .reshape((1, 1, self.config.vit_d_model))?
            .broadcast_as((batch, 1, self.config.vit_d_model))?;
        x = Tensor::cat(&[&cls, &x], 1)?;
        let pos = self.position_embedding.reshape((
            1,
            self.config.num_patches + 1,
            self.config.vit_d_model,
        ))?;
        x = x.broadcast_add(&pos)?;
        x = self.pre_norm.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.post_norm.forward(&x)?;
        Ok(x.narrow(1, 1, self.config.num_patches)?)
    }

    fn patchify(&self, images: &Tensor) -> Result<Tensor> {
        let dims = images.dims();
        if dims.len() != 4 {
            return Err(AarambhError::Shape(format!(
                "images must have shape [batch, channels, height, width], got {dims:?}"
            )));
        }
        let (batch, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        if channels != self.config.in_channels
            || height != self.config.image_size
            || width != self.config.image_size
        {
            return Err(AarambhError::Shape(format!(
                "images must be [batch, {}, {}, {}], got {dims:?}",
                self.config.in_channels, self.config.image_size, self.config.image_size
            )));
        }

        let patch = self.config.patch_size;
        let grid = self.config.image_size / patch;
        let patches = images.unfold(2, patch, patch)?.unfold(3, patch, patch)?;
        let patches = patches.permute((0, 2, 3, 1, 4, 5))?.contiguous()?;
        Ok(patches.reshape((batch, grid * grid, self.config.patch_dim()))?)
    }
}

fn normalize_clip_tensors(
    source: HashMap<String, Tensor>,
    config: &VisionEncoderConfig,
) -> Result<HashMap<String, Tensor>> {
    if source.contains_key("patch_embed.weight") {
        return Ok(source);
    }

    let mut out = HashMap::new();
    out.insert(
        "patch_embed.weight".to_string(),
        hf_patch_weight(&source, config)?,
    );
    insert_first(
        &mut out,
        &source,
        "class_embedding",
        &[
            "vision_model.embeddings.class_embedding",
            "embeddings.class_embedding",
        ],
    )?;
    insert_first(
        &mut out,
        &source,
        "position_embedding",
        &[
            "vision_model.embeddings.position_embedding.weight",
            "embeddings.position_embedding.weight",
        ],
    )?;
    insert_first(
        &mut out,
        &source,
        "pre_norm.weight",
        &[
            "vision_model.pre_layrnorm.weight",
            "vision_model.pre_layernorm.weight",
            "pre_layrnorm.weight",
            "pre_layernorm.weight",
        ],
    )?;
    insert_first(
        &mut out,
        &source,
        "pre_norm.bias",
        &[
            "vision_model.pre_layrnorm.bias",
            "vision_model.pre_layernorm.bias",
            "pre_layrnorm.bias",
            "pre_layernorm.bias",
        ],
    )?;
    insert_first(
        &mut out,
        &source,
        "post_norm.weight",
        &[
            "vision_model.post_layernorm.weight",
            "post_layernorm.weight",
        ],
    )?;
    insert_first(
        &mut out,
        &source,
        "post_norm.bias",
        &["vision_model.post_layernorm.bias", "post_layernorm.bias"],
    )?;

    for layer in 0..config.vit_layers {
        let dst = format!("blocks.{layer}");
        let hf = format!("vision_model.encoder.layers.{layer}");
        let short = format!("encoder.layers.{layer}");
        for (canonical, hf_suffix) in [
            ("norm1.weight", "layer_norm1.weight"),
            ("norm1.bias", "layer_norm1.bias"),
            ("attn.q_proj.weight", "self_attn.q_proj.weight"),
            ("attn.q_proj.bias", "self_attn.q_proj.bias"),
            ("attn.k_proj.weight", "self_attn.k_proj.weight"),
            ("attn.k_proj.bias", "self_attn.k_proj.bias"),
            ("attn.v_proj.weight", "self_attn.v_proj.weight"),
            ("attn.v_proj.bias", "self_attn.v_proj.bias"),
            ("attn.out_proj.weight", "self_attn.out_proj.weight"),
            ("attn.out_proj.bias", "self_attn.out_proj.bias"),
            ("norm2.weight", "layer_norm2.weight"),
            ("norm2.bias", "layer_norm2.bias"),
            ("mlp.fc1.weight", "mlp.fc1.weight"),
            ("mlp.fc1.bias", "mlp.fc1.bias"),
            ("mlp.fc2.weight", "mlp.fc2.weight"),
            ("mlp.fc2.bias", "mlp.fc2.bias"),
        ] {
            insert_first(
                &mut out,
                &source,
                &format!("{dst}.{canonical}"),
                &[
                    &format!("{hf}.{hf_suffix}"),
                    &format!("{short}.{hf_suffix}"),
                ],
            )?;
        }
    }

    Ok(out)
}

fn hf_patch_weight(
    source: &HashMap<String, Tensor>,
    config: &VisionEncoderConfig,
) -> Result<Tensor> {
    let weight = first_tensor(
        source,
        &[
            "vision_model.embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.weight",
        ],
    )?;
    match weight.dims() {
        [out, channels, height, width]
            if *out == config.vit_d_model
                && *channels == config.in_channels
                && *height == config.patch_size
                && *width == config.patch_size =>
        {
            Ok(weight.reshape((config.vit_d_model, config.patch_dim()))?)
        }
        dims if dims == [config.vit_d_model, config.patch_dim()] => Ok(weight.clone()),
        dims => Err(AarambhError::Shape(format!(
            "CLIP patch embedding has shape {dims:?}, expected [{}, {}, {}, {}]",
            config.vit_d_model, config.in_channels, config.patch_size, config.patch_size
        ))),
    }
}

fn insert_first(
    out: &mut HashMap<String, Tensor>,
    source: &HashMap<String, Tensor>,
    canonical: &str,
    candidates: &[&str],
) -> Result<()> {
    out.insert(
        canonical.to_string(),
        first_tensor(source, candidates)?.clone(),
    );
    Ok(())
}

fn first_tensor<'a>(
    source: &'a HashMap<String, Tensor>,
    candidates: &[&str],
) -> Result<&'a Tensor> {
    candidates
        .iter()
        .find_map(|name| source.get(*name))
        .ok_or_else(|| {
            AarambhError::Checkpoint(format!(
                "missing CLIP tensor; tried {}",
                candidates.join(", ")
            ))
        })
}

#[derive(Debug, Clone)]
struct VisionBlock {
    norm1: LayerNorm,
    attn: VisionAttention,
    norm2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(config: &VisionEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let norm1 = layer_norm(config.vit_d_model, config.norm_eps, vb.pp("norm1"))?;
        let attn = VisionAttention::new(config, vb.pp("attn"))?;
        let norm2 = layer_norm(config.vit_d_model, config.norm_eps, vb.pp("norm2"))?;
        let mlp = VisionMlp::new(config, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (residual + x)?;
        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

#[derive(Debug, Clone)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(config: &VisionEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let q_proj = linear(config.vit_d_model, config.vit_d_model, vb.pp("q_proj"))?;
        let k_proj = linear(config.vit_d_model, config.vit_d_model, vb.pp("k_proj"))?;
        let v_proj = linear(config.vit_d_model, config.vit_d_model, vb.pp("v_proj"))?;
        let out_proj = linear(config.vit_d_model, config.vit_d_model, vb.pp("out_proj"))?;
        let head_dim = config.head_dim();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            heads: config.vit_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (batch, seq_len, width) = (dims[0], dims[1], dims[2]);
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let attn = q.matmul(&k.transpose(2, 3)?)?.affine(self.scale, 0.0)?;
        let attn = softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?;
        let out = out.reshape((batch, seq_len, width))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn new(config: &VisionEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let fc1 = linear(config.vit_d_model, config.mlp_dim, vb.pp("fc1"))?;
        let fc2 = linear(config.mlp_dim, config.vit_d_model, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.gelu()?;
        Ok(self.fc2.forward(&x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_config() -> VisionEncoderConfig {
        VisionEncoderConfig {
            patch_size: 4,
            image_size: 8,
            in_channels: 3,
            vit_d_model: 8,
            vit_layers: 1,
            vit_heads: 2,
            mlp_dim: 16,
            num_patches: 4,
            norm_eps: 1e-5,
        }
    }

    #[test]
    fn vision_encoder_outputs_patch_tokens() {
        let device = Device::Cpu;
        let config = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = ClipVisionEncoder::new(config, vb).unwrap();
        let image = Tensor::zeros((1, 3, 8, 8), DType::F32, &device).unwrap();
        let output = encoder.forward(&image).unwrap();
        assert_eq!(output.dims(), &[1, 4, 8]);
    }
}
