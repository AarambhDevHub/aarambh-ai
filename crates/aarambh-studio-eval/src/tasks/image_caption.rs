use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use aarambh_studio_tokenizer::{IMAGE, IMAGE_END, IMAGE_ID};
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_vision::{
    ClipVisionEncoder, ImagePreprocessor, ProjectorConfig, VisionEncoderConfig, VisionModel,
    VisionPreprocessConfig, VisionProjector, interleave_image_tokens,
};
use candle_core::Tensor;
use serde::Deserialize;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::{first_existing, read_jsonl};

#[derive(Debug, Clone, Deserialize)]
struct ImageCaptionExample {
    image: Option<PathBuf>,
    image_path: Option<PathBuf>,
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    expected_keywords: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
}

/// Image-captioning smoke task for Phase 19 vision projector checkpoints.
pub struct ImageCaptionTask;

impl EvalTask for ImageCaptionTask {
    fn name(&self) -> &'static str {
        "image-caption"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let data_path = first_existing(&[
            config.data_dir.join("image_caption").join("data.jsonl"),
            config
                .data_dir
                .join("image_caption_smoke")
                .join("data.jsonl"),
            config.data_dir.join("image_caption.jsonl"),
        ])?;
        let examples = read_jsonl::<ImageCaptionExample>(&data_path, config.max_examples)?;
        let runtime = load_vision_runtime(context, config)?;
        let image_root = data_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| config.data_dir.clone());
        let mut passed = 0usize;

        for example in &examples {
            let image_path = resolve_image_path(&image_root, example)?;
            let prompt = ensure_image_prompt(&example.prompt);
            let embeddings = build_prompt_embeddings(context, &runtime, &image_path, &prompt)?;
            let output =
                greedy_generate_from_embeddings(context, &embeddings, config.max_new_tokens)?;
            if matches_expected(&output, &example.expected_keywords) {
                passed += 1;
            }
        }

        let mut score = TaskScore::accuracy("image-caption", passed, examples.len());
        score.metric = "smoke_pass_rate".into();
        Ok(score)
    }
}

fn load_vision_runtime(context: &EvalContext, config: &EvalConfig) -> Result<VisionModel> {
    let config_path = config
        .config_path
        .as_ref()
        .ok_or_else(|| AarambhError::Config("image-caption eval requires --config".into()))?;
    let run_config = TrainingRunConfig::from_toml(config_path)?;
    let vision = run_config.vision.as_ref().ok_or_else(|| {
        AarambhError::Config("image-caption eval requires [vision] config".into())
    })?;
    let encoder_config = VisionEncoderConfig::from_json(&vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &vision.clip_weights_path,
        encoder_config.clone(),
        context.device(),
        context.dtype(),
    )?;
    let projector_path = match &vision.projector_path {
        Some(path) => path.clone(),
        None => default_projector_path(&run_config.train.checkpoint_dir)?,
    };
    let projector_config = ProjectorConfig {
        vit_d_model: encoder_config.vit_d_model,
        llm_d_model: run_config.model.hidden_dim,
        hidden_mult: vision.projector_hidden_mult,
    };
    let projector = VisionProjector::load_safetensors(
        projector_path,
        projector_config,
        context.device(),
        context.dtype(),
    )?;
    Ok(VisionModel::new(encoder, projector))
}

fn default_projector_path(checkpoint_dir: &Path) -> Result<PathBuf> {
    for pointer_name in ["best.json", "latest.json"] {
        let pointer_path = checkpoint_dir.join(pointer_name);
        if pointer_path.exists() {
            let pointer: CheckpointPointer =
                serde_json::from_reader(fs::File::open(pointer_path)?)?;
            return Ok(pointer.path.join("model.safetensors"));
        }
    }
    Err(AarambhError::Config(format!(
        "no projector path configured and no best/latest pointer found in {}",
        checkpoint_dir.display()
    )))
}

fn build_prompt_embeddings(
    context: &EvalContext,
    vision: &VisionModel,
    image_path: &Path,
    prompt: &str,
) -> Result<Tensor> {
    context.tokenizer().validate_vision_special_tokens()?;
    let prompt_ids = context.tokenizer().encode(prompt)?;
    let text = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
    let text_embeddings = context.model().embed_tokens(&text)?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: vision.encoder().config().image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let image = preprocess
        .preprocess_path(image_path, context.device())?
        .unsqueeze(0)?;
    let image_embeddings = vision.forward(&image)?;
    interleave_image_tokens(&prompt_ids, &text_embeddings, &image_embeddings, IMAGE_ID)
}

fn greedy_generate_from_embeddings(
    context: &EvalContext,
    embeddings: &Tensor,
    max_new_tokens: usize,
) -> Result<String> {
    let prompt_len = embeddings.dims()[1];
    if prompt_len >= context.max_seq_len() {
        return Err(AarambhError::Shape(format!(
            "vision prompt length {prompt_len} leaves no room in max_seq_len {}",
            context.max_seq_len()
        )));
    }
    let budget = max_new_tokens.min(context.max_seq_len() - prompt_len);
    let mut caches = context.model().empty_kv_cache();
    let logits = context
        .model()
        .forward_embeddings_with_cache(embeddings, 0, &mut caches)?;
    let mut next_logits = last_logits(&logits)?;
    let mut generated = Vec::with_capacity(budget);

    for step in 0..budget {
        let logits_vec = next_logits.to_vec1::<f32>()?;
        let token_id = argmax(&logits_vec) as u32;
        if token_id == context.tokenizer().eos_token_id() {
            break;
        }
        generated.push(token_id);
        context.record_context_len(prompt_len + generated.len());
        if step + 1 == budget {
            break;
        }
        let offset = prompt_len + generated.len() - 1;
        let input = Tensor::from_vec(vec![token_id], (1, 1), context.device())?;
        let logits = context
            .model()
            .forward_with_cache(&input, offset, &mut caches)?;
        next_logits = last_logits(&logits)?;
    }

    context.tokenizer().decode(&generated)
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 || dims[1] == 0 {
        return Err(AarambhError::Shape(format!(
            "expected logits [batch, seq, vocab], got {dims:?}"
        )));
    }
    Ok(logits.narrow(1, dims[1] - 1, 1)?.squeeze(1)?.squeeze(0)?)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn ensure_image_prompt(prompt: &str) -> String {
    if prompt.contains(IMAGE) {
        prompt.to_string()
    } else {
        format!("{IMAGE}{IMAGE_END}\n{prompt}")
    }
}

fn resolve_image_path(root: &Path, example: &ImageCaptionExample) -> Result<PathBuf> {
    let path = example
        .image_path
        .as_ref()
        .or(example.image.as_ref())
        .ok_or_else(|| AarambhError::Config("image-caption example is missing image".into()))?;
    if path.is_absolute() {
        Ok(path.clone())
    } else {
        Ok(root.join(path))
    }
}

fn matches_expected(output: &str, keywords: &[String]) -> bool {
    let output = output.trim();
    if output.is_empty() {
        return false;
    }
    if keywords.is_empty() {
        return true;
    }
    let output_lower = output.to_ascii_lowercase();
    keywords
        .iter()
        .any(|keyword| output_lower.contains(&keyword.to_ascii_lowercase()))
}

fn default_prompt() -> String {
    "Describe this image.".to_string()
}
