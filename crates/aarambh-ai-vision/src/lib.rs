//! Vision encoder, projector, preprocessing, and multimodal fusion utilities.
#![deny(missing_docs)]

/// CLIP-style frozen vision encoder.
pub mod encoder;
/// LLaVA-style image token fusion helpers.
pub mod fusion;
/// Image decode, resize, crop, and normalization.
pub mod preprocess;
/// Trainable projector from vision width to language-model width.
pub mod projector;

pub use encoder::{ClipVisionEncoder, VisionEncoderConfig};
pub use fusion::interleave_image_tokens;
pub use preprocess::{ImagePreprocessor, VisionPreprocessConfig};
pub use projector::{ProjectorConfig, VisionProjector};

/// Frozen vision encoder plus trainable language-model projector.
#[derive(Debug, Clone)]
pub struct VisionModel {
    encoder: ClipVisionEncoder,
    projector: VisionProjector,
}

impl VisionModel {
    /// Create a vision model from an encoder and projector.
    pub fn new(encoder: ClipVisionEncoder, projector: VisionProjector) -> Self {
        Self { encoder, projector }
    }

    /// Return the frozen image encoder.
    pub fn encoder(&self) -> &ClipVisionEncoder {
        &self.encoder
    }

    /// Return the trainable projector.
    pub fn projector(&self) -> &VisionProjector {
        &self.projector
    }

    /// Encode an image tensor and project patch tokens into language-model width.
    pub fn forward(
        &self,
        image: &candle_core::Tensor,
    ) -> aarambh_ai_core::Result<candle_core::Tensor> {
        let patch_tokens = self.encoder.forward(image)?;
        self.projector.forward(&patch_tokens)
    }
}
