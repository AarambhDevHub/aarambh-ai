use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
use aarambh_ai_tokenizer::{FRAME_SEP, FRAME_SEP_ID, VIDEO, VIDEO_END, VIDEO_ID};
use aarambh_ai_train::TrainingRunConfig;
use aarambh_ai_vision::{
    ImagePreprocessor, TemporalEncoder, TemporalEncodingConfig, TemporalEncodingKind,
    VideoSamplingConfig, VisionPreprocessConfig, decode_sampled_video, interleave_video_tokens,
    load_video_qa,
};
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::first_existing;
use crate::tasks::vqa::{greedy_generate_from_embeddings, load_vision_runtime};

/// NExT-QA and normalized video question-answering evaluation task.
pub struct VideoQaTask;

impl EvalTask for VideoQaTask {
    fn name(&self) -> &'static str {
        "video-qa"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let data_path = first_existing(&[
            config.data_dir.join("nextqa").join("test.csv"),
            config.data_dir.join("video_qa").join("data.jsonl"),
            config.data_dir.join("video_qa_smoke").join("data.jsonl"),
            config.data_dir.join("video_qa.jsonl"),
        ])?;
        let examples = load_video_qa(&data_path, config.max_examples)?;
        let config_path = config
            .config_path
            .as_ref()
            .ok_or_else(|| AarambhError::Config("video QA eval requires --config".into()))?;
        let run_config = TrainingRunConfig::from_toml(config_path)?;
        let video_config = run_config
            .vision
            .as_ref()
            .and_then(|vision| vision.video.as_ref())
            .ok_or_else(|| AarambhError::Config("video QA eval requires [vision.video]".into()))?;
        video_config.validate()?;
        context.tokenizer().validate_video_special_tokens()?;
        let vision = load_vision_runtime(context, config)?;
        let temporal_varmap = VarMap::new();
        let temporal_vb =
            VarBuilder::from_varmap(&temporal_varmap, context.dtype(), context.device());
        let temporal = TemporalEncoder::new(
            TemporalEncodingConfig {
                max_frames: video_config.max_frame_count,
                hidden_dim: vision.encoder().config().vit_d_model,
                kind: video_config.temporal_encoding,
            },
            (video_config.temporal_encoding == TemporalEncodingKind::Learned)
                .then_some(temporal_vb),
        )?;
        if video_config.temporal_encoding == TemporalEncodingKind::Learned {
            let path = video_config.temporal_path.as_ref().ok_or_else(|| {
                AarambhError::Config(
                    "video QA eval with learned positions requires vision.video.temporal_path"
                        .into(),
                )
            })?;
            let mut temporal_varmap = temporal_varmap;
            temporal_varmap.load(path)?;
        }
        let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
            image_size: vision.encoder().config().image_size,
            ..VisionPreprocessConfig::default()
        })?;
        let data_root = data_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| config.data_dir.clone());
        let sampling = VideoSamplingConfig {
            frame_count: video_config.frame_count,
            max_frame_count: video_config.max_frame_count,
            strategy: video_config.sampling,
            scene_min_gap: video_config.scene_min_gap,
        };
        let mut passed = 0usize;
        for example in &examples {
            let path =
                resolve_video_path(&video_config.video_root, &data_root, &example.video_path);
            let prompt = video_prompt(
                &format!("<|user|>\n{}\n<|assistant|>\n", example.question),
                sampling.frame_count,
            );
            let prompt_ids = context.tokenizer().encode(&prompt)?;
            let text =
                Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
            let text_embeddings = context.model().embed_tokens(&text)?;
            let sampled = decode_sampled_video(&path, &sampling)?;
            let pixels = preprocess.preprocess_rgb_batch(&sampled.frames, context.device())?;
            let mut chunks = Vec::new();
            for start in (0..sampled.frames.len()).step_by(video_config.encoder_frame_batch_size) {
                let len = video_config
                    .encoder_frame_batch_size
                    .min(sampled.frames.len() - start);
                chunks.push(vision.encoder().forward(&pixels.narrow(0, start, len)?)?);
            }
            let refs = chunks.iter().collect::<Vec<_>>();
            let encoded = Tensor::cat(&refs, 0)?;
            let projected = vision.projector().forward(&temporal.forward(&encoded)?)?;
            let embeddings = interleave_video_tokens(
                &prompt_ids,
                &text_embeddings,
                &projected,
                VIDEO_ID,
                FRAME_SEP_ID,
            )?;
            let output =
                greedy_generate_from_embeddings(context, &embeddings, config.max_new_tokens)?;
            if normalize_answer(&output) == normalize_answer(&example.answer) {
                passed += 1;
            }
        }
        Ok(TaskScore::accuracy("video-qa", passed, examples.len()))
    }
}

fn video_prompt(prompt: &str, frames: usize) -> String {
    let mut marker = String::from(VIDEO);
    for _ in 1..frames {
        marker.push_str(FRAME_SEP);
    }
    marker.push_str(VIDEO_END);
    format!("{marker}\n{prompt}")
}

fn resolve_video_path(config_root: &Path, data_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let configured = config_root.join(path);
    if configured.exists() {
        configured
    } else {
        data_root.join(path)
    }
}

fn normalize_answer(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_contains_exact_frame_separator_count() {
        let prompt = video_prompt("question", 4);
        assert_eq!(prompt.matches(FRAME_SEP).count(), 3);
    }

    #[test]
    fn answer_normalization_handles_option_punctuation() {
        assert_eq!(normalize_answer(" A.\n"), "a");
    }
}
