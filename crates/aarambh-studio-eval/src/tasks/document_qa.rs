use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use aarambh_studio_tokenizer::{DOCUMENT, DOCUMENT_END, DOCUMENT_ID, PAGE_SEP, PAGE_SEP_ID};
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_vision::{
    DocumentSource, ImagePreprocessor, LayoutAwareProjector, LayoutEncodingKind,
    LayoutProjectorConfig, PageRasterizer, PageRasterizerConfig, VisionPreprocessConfig,
    interleave_document_tokens, load_document_qa_jsonl,
};
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::first_existing;
use crate::tasks::vqa::{greedy_generate_from_embeddings, load_vision_runtime};

/// DocVQA-style PDF and scanned-document question-answering task.
pub struct DocumentQaTask;

impl EvalTask for DocumentQaTask {
    fn name(&self) -> &'static str {
        "document-qa"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let data_path = first_existing(&[
            config.data_dir.join("docvqa").join("data.jsonl"),
            config.data_dir.join("document_qa").join("data.jsonl"),
            config.data_dir.join("document_qa_smoke").join("data.jsonl"),
            config.data_dir.join("document_qa.jsonl"),
        ])?;
        let examples = load_document_qa_jsonl(&data_path, config.max_examples)?;
        let config_path = config
            .config_path
            .as_ref()
            .ok_or_else(|| AarambhError::Config("document QA eval requires --config".into()))?;
        let run_config = TrainingRunConfig::from_toml(config_path)?;
        let document = run_config
            .vision
            .as_ref()
            .and_then(|vision| vision.document.as_ref())
            .ok_or_else(|| {
                AarambhError::Config("document QA eval requires [vision.document]".into())
            })?;
        document.validate()?;
        context.tokenizer().validate_document_special_tokens()?;

        let vision = load_vision_runtime(context, config)?;
        let encoder_config = vision.encoder().config();
        let patch_side = encoder_config.image_size / encoder_config.patch_size;
        let layout_varmap = VarMap::new();
        let layout_vb = VarBuilder::from_varmap(&layout_varmap, context.dtype(), context.device());
        let layout = LayoutAwareProjector::new(
            vision.projector().clone(),
            LayoutProjectorConfig {
                patch_rows: patch_side,
                patch_cols: patch_side,
                hidden_dim: run_config.model.hidden_dim,
                encoding: document.layout_encoding,
            },
            (document.layout_encoding == LayoutEncodingKind::Learned).then_some(layout_vb),
        )?;
        if document.layout_encoding == LayoutEncodingKind::Learned {
            let path = document.layout_path.as_ref().ok_or_else(|| {
                AarambhError::Config(
                    "document QA eval with learned layout requires vision.document.layout_path"
                        .into(),
                )
            })?;
            let mut layout_varmap = layout_varmap;
            layout_varmap.load(path)?;
        }
        let rasterizer = PageRasterizer::new(PageRasterizerConfig {
            target_dpi: document.target_dpi,
            max_pages_per_document: document.max_pages_per_document,
            max_page_pixels: document.max_page_pixels,
        })?;
        let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
            image_size: encoder_config.image_size,
            ..VisionPreprocessConfig::default()
        })?;
        let data_root = data_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| config.data_dir.clone());

        let mut anls_sum = 0.0;
        let mut exact = 0usize;
        let mut table_anls_sum = 0.0;
        let mut table_examples = 0usize;
        for example in &examples {
            let source =
                resolve_document_source(&document.document_root, &data_root, &example.source());
            let rendered = rasterizer.rasterize(&source, example.pages.as_deref())?;
            let page_count = rendered.pages.len();
            let pages = rendered
                .pages
                .into_iter()
                .map(|page| page.image)
                .collect::<Vec<_>>();
            let pixels = preprocess.preprocess_document_pages(&pages, context.device())?;
            let mut chunks = Vec::new();
            for start in (0..pages.len()).step_by(document.encoder_page_batch_size) {
                let len = document.encoder_page_batch_size.min(pages.len() - start);
                chunks.push(vision.encoder().forward(&pixels.narrow(0, start, len)?)?);
            }
            let references = chunks.iter().collect::<Vec<_>>();
            let encoded = Tensor::cat(&references, 0)?;
            let projected = layout.forward(&encoded, (patch_side, patch_side))?;
            let prompt = document_prompt(
                &format!("<|user|>\n{}\n<|assistant|>\n", example.question),
                page_count,
            );
            let prompt_ids = context.tokenizer().encode(&prompt)?;
            let text =
                Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
            let text_embeddings = context.model().embed_tokens(&text)?;
            let embeddings = interleave_document_tokens(
                &prompt_ids,
                &text_embeddings,
                &projected,
                DOCUMENT_ID,
                PAGE_SEP_ID,
            )?;
            let output =
                greedy_generate_from_embeddings(context, &embeddings, config.max_new_tokens)?;
            let score = example_anls(&output, &example.answers);
            anls_sum += score;
            exact += usize::from(
                example
                    .answers
                    .iter()
                    .any(|answer| normalize_answer(&output) == normalize_answer(answer)),
            );
            if example
                .tags
                .iter()
                .any(|tag| tag.eq_ignore_ascii_case("table"))
            {
                table_anls_sum += score;
                table_examples += 1;
            }
        }

        let count = examples.len();
        let mut score = TaskScore {
            name: "document-qa".into(),
            metric: "anls".into(),
            value: anls_sum / count as f64,
            higher_is_better: true,
            examples: count,
            correct: Some(exact),
            loss: None,
            ppl: None,
            details: Default::default(),
        }
        .with_detail("exact_match", exact as f64 / count as f64);
        if table_examples > 0 {
            score = score
                .with_detail("table_anls", table_anls_sum / table_examples as f64)
                .with_detail("table_examples", table_examples as f64);
        }
        Ok(score)
    }
}

fn document_prompt(prompt: &str, pages: usize) -> String {
    let mut marker = String::from(DOCUMENT);
    for _ in 1..pages {
        marker.push_str(PAGE_SEP);
    }
    marker.push_str(DOCUMENT_END);
    format!("{marker}\n{prompt}")
}

fn resolve_document_source(
    config_root: &Path,
    data_root: &Path,
    source: &DocumentSource,
) -> DocumentSource {
    let resolve = |path: &Path| -> PathBuf {
        if path.is_absolute() {
            return path.to_path_buf();
        }
        let configured = config_root.join(path);
        if configured.exists() {
            configured
        } else {
            data_root.join(path)
        }
    };
    match source {
        DocumentSource::File(path) => DocumentSource::File(resolve(path)),
        DocumentSource::PageImages(paths) => {
            DocumentSource::PageImages(paths.iter().map(|path| resolve(path)).collect())
        }
    }
}

fn example_anls(prediction: &str, answers: &[String]) -> f64 {
    let prediction = normalize_answer(prediction);
    answers
        .iter()
        .map(|answer| normalized_levenshtein(&prediction, &normalize_answer(answer)))
        .map(|similarity| if similarity >= 0.5 { similarity } else { 0.0 })
        .fold(0.0, f64::max)
}

fn normalized_levenshtein(left: &str, right: &str) -> f64 {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let denominator = left.len().max(right.len());
    if denominator == 0 {
        return 1.0;
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0usize; right.len() + 1];
    for (row, left_char) in left.iter().enumerate() {
        current[0] = row + 1;
        for (column, right_char) in right.iter().enumerate() {
            let substitution = previous[column] + usize::from(left_char != right_char);
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    1.0 - previous[right.len()] as f64 / denominator as f64
}

fn normalize_answer(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anls_uses_best_reference_and_threshold() {
        assert_eq!(
            example_anls("invoice 42", &["wrong".into(), "Invoice 42".into()]),
            1.0
        );
        assert_eq!(example_anls("abc", &["xyz".into()]), 0.0);
    }

    #[test]
    fn marker_has_one_separator_per_additional_page() {
        assert_eq!(document_prompt("q", 4).matches(PAGE_SEP).count(), 3);
    }
}
