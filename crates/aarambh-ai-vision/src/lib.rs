//! Vision encoder, projector, preprocessing, and multimodal fusion utilities.
#![deny(missing_docs)]

/// PDF and scanned-page rasterization plus frozen-feature caching.
pub mod document_sample;
/// CLIP-style frozen vision encoder.
pub mod encoder;
/// LLaVA-style image token fusion helpers.
pub mod fusion;
/// Vision-language instruction data loading.
pub mod instruct_data;
/// Layout-aware projection for document page patches.
pub mod layout_projector;
/// Image decode, resize, crop, and normalization.
pub mod preprocess;
/// Trainable projector from vision width to language-model width.
pub mod projector;
/// Temporal positions for sampled video frames.
pub mod temporal;
/// Native H.264 MP4 decode, sampling, and frozen-feature caching.
pub mod video;
/// Video instruction data and NExT-QA normalization.
pub mod video_data;
/// Video placeholder and frame-separator fusion helpers.
pub mod video_fusion;

pub use document_sample::{
    DocumentFeatureCache, DocumentFeatureCacheKey, DocumentSource, PageRasterizer,
    PageRasterizerConfig, RasterizedDocument, RasterizedPage,
};
pub use encoder::{ClipVisionEncoder, VisionEncoderConfig};
pub use fusion::interleave_image_tokens;
pub use instruct_data::{DocQaExample, VqaExample, load_document_qa_jsonl, load_vqa_jsonl};
pub use layout_projector::{LayoutAwareProjector, LayoutEncodingKind, LayoutProjectorConfig};
pub use preprocess::{ImagePreprocessor, VisionPreprocessConfig};
pub use projector::{ProjectorConfig, VisionProjector};
pub use temporal::{TemporalEncoder, TemporalEncodingConfig, TemporalEncodingKind};
pub use video::{
    FrameSamplingStrategy, SampledVideo, VideoFeatureCache, VideoFeatureCacheKey,
    VideoSamplingConfig, decode_sampled_video, scene_aware_frame_indices, uniform_frame_indices,
};
pub use video_data::{VideoQaExample, load_video_qa};
pub use video_fusion::{
    interleave_document_tokens, interleave_media_sequence_tokens, interleave_video_tokens,
};

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
