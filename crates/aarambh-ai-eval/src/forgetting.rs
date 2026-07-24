use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_ai_core::{AarambhError, Configurable, Result, TokenizerLike};
use aarambh_ai_tokenizer::BpeTokenizer;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::harness::{EvalConfig, EvalContext, run_all};
use crate::report::TaskScore;

/// Default absolute score change treated as statistically meaningful.
pub const DEFAULT_SIGNIFICANCE_THRESHOLD: f64 = 0.02;

/// Stable capability categories tracked by Phase 38 probes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Capability {
    /// Mathematical problem solving.
    Math,
    /// Source-code generation.
    Code,
    /// Commonsense and multi-step reasoning.
    Reasoning,
    /// Factual knowledge.
    Factual,
    /// Static-image understanding.
    Vision,
    /// Temporal video understanding.
    Video,
    /// Document and layout understanding.
    Document,
    /// Function calling and long-horizon tool use.
    ToolUse,
}

impl Capability {
    /// Return all Phase 38 capabilities in report order.
    pub const fn all() -> [Self; 8] {
        [
            Self::Math,
            Self::Code,
            Self::Reasoning,
            Self::Factual,
            Self::Vision,
            Self::Video,
            Self::Document,
            Self::ToolUse,
        ]
    }

    /// Return the eval tasks allowed for this capability.
    pub const fn allowed_tasks(self) -> &'static [&'static str] {
        match self {
            Self::Math => &["gsm8k"],
            Self::Code => &["humaneval"],
            Self::Reasoning => &["hellaswag"],
            Self::Factual => &["mmlu"],
            Self::Vision => &["vqa"],
            Self::Video => &["video-qa"],
            Self::Document => &["document-qa"],
            Self::ToolUse => &["tool-calling", "tool-chain"],
        }
    }
}

impl Display for Capability {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Math => "math",
            Self::Code => "code",
            Self::Reasoning => "reasoning",
            Self::Factual => "factual",
            Self::Vision => "vision",
            Self::Video => "video",
            Self::Document => "document",
            Self::ToolUse => "tool-use",
        };
        formatter.write_str(value)
    }
}

impl FromStr for Capability {
    type Err = AarambhError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "math" => Ok(Self::Math),
            "code" => Ok(Self::Code),
            "reasoning" => Ok(Self::Reasoning),
            "factual" => Ok(Self::Factual),
            "vision" => Ok(Self::Vision),
            "video" => Ok(Self::Video),
            "document" => Ok(Self::Document),
            "tool-use" | "tool_use" | "tools" => Ok(Self::ToolUse),
            other => Err(AarambhError::Config(format!(
                "unknown forgetting capability '{other}'"
            ))),
        }
    }
}

/// One capability probe composed from existing evaluation task subsets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityProbe {
    /// Capability represented by this probe.
    pub capability: Capability,
    /// Existing eval task selectors used to score the capability.
    pub tasks: Vec<String>,
    /// Optional per-task example cap combined with the run-level cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_examples: Option<usize>,
}

/// Versioned set of fixed capability probes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeManifest {
    /// Probe-manifest schema version.
    pub schema_version: u32,
    /// Stable suite identifier used to separate incompatible curves.
    pub suite_id: String,
    /// Capability probes in deterministic execution order.
    pub probes: Vec<CapabilityProbe>,
}

impl ProbeManifest {
    /// Load and validate a JSON probe manifest.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = fs::File::open(path)?;
        let manifest: Self = serde_json::from_reader(file)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate schema, capability uniqueness, task ownership, and limits.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(AarambhError::Config(format!(
                "unsupported forgetting probe schema {}, expected 1",
                self.schema_version
            )));
        }
        validate_id("forgetting suite_id", &self.suite_id)?;
        if self.probes.is_empty() {
            return Err(AarambhError::Config(
                "forgetting probe manifest must contain at least one probe".into(),
            ));
        }
        let mut capabilities = BTreeSet::new();
        for probe in &self.probes {
            if !capabilities.insert(probe.capability) {
                return Err(AarambhError::Config(format!(
                    "duplicate forgetting capability '{}'",
                    probe.capability
                )));
            }
            if probe.tasks.is_empty() {
                return Err(AarambhError::Config(format!(
                    "forgetting capability '{}' has no eval tasks",
                    probe.capability
                )));
            }
            if probe.max_examples == Some(0) {
                return Err(AarambhError::Config(format!(
                    "forgetting capability '{}' max_examples must be non-zero",
                    probe.capability
                )));
            }
            let allowed = probe.capability.allowed_tasks();
            let mut tasks = BTreeSet::new();
            for task in &probe.tasks {
                let normalized = normalize_task(task);
                if !allowed.contains(&normalized.as_str()) {
                    return Err(AarambhError::Config(format!(
                        "eval task '{task}' is not valid for forgetting capability '{}'",
                        probe.capability
                    )));
                }
                if !tasks.insert(normalized) {
                    return Err(AarambhError::Config(format!(
                        "duplicate eval task '{task}' for forgetting capability '{}'",
                        probe.capability
                    )));
                }
            }
        }
        Ok(())
    }

    /// Return a deterministic SHA-256 fingerprint of the validated manifest.
    pub fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

/// One successful capability score from a probe run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapabilityScore {
    /// Scored capability.
    pub capability: Capability,
    /// Example-weighted aggregate score.
    pub score: f64,
    /// Total examples contributing to the aggregate.
    pub examples: usize,
    /// Underlying eval task scores.
    pub tasks: Vec<TaskScore>,
    /// Optional per-example MoE routing signatures.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routing: Vec<RoutingSignature>,
}

/// One capability skipped because its data, permission, or modality was unavailable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProbeSkip {
    /// Skipped capability.
    pub capability: Capability,
    /// Human-readable reason.
    pub reason: String,
}

/// Routed expert sets captured for one probe example.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingSignature {
    /// Stable example identifier within the capability suite.
    pub example_id: String,
    /// Sorted routed expert indices, grouped by MoE layer.
    pub experts_by_layer: Vec<Vec<usize>>,
}

/// Routing change between baseline and current probe runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingDrift {
    /// Capability whose routes were compared.
    pub capability: Capability,
    /// Fraction of comparable examples with a changed routed-expert set.
    pub drift_rate: f64,
    /// Number of examples present in both runs.
    pub compared_examples: usize,
    /// Number of compared examples whose routes changed.
    pub changed_examples: usize,
}

/// Scores and skips produced by one complete probe suite execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingRun {
    /// Checkpoint or self-learning session identifier.
    pub checkpoint_or_session_id: String,
    /// Probe suite identifier.
    pub suite_id: String,
    /// Probe manifest fingerprint.
    pub manifest_sha256: String,
    /// Optional tokenizer fingerprint supplied by the caller.
    pub tokenizer_sha256: Option<String>,
    /// UNIX timestamp in seconds.
    pub timestamp_unix: u64,
    /// Successful capability scores.
    pub scores: Vec<CapabilityScore>,
    /// Explicitly skipped capabilities.
    pub skipped: Vec<ProbeSkip>,
}

impl ForgettingRun {
    /// Build a validated probe run.
    pub fn new(
        checkpoint_or_session_id: impl Into<String>,
        manifest: &ProbeManifest,
        tokenizer_sha256: Option<String>,
        scores: Vec<CapabilityScore>,
        skipped: Vec<ProbeSkip>,
    ) -> Result<Self> {
        let checkpoint_or_session_id = checkpoint_or_session_id.into();
        validate_id("checkpoint_or_session_id", &checkpoint_or_session_id)?;
        validate_scores(&scores)?;
        Ok(Self {
            checkpoint_or_session_id,
            suite_id: manifest.suite_id.clone(),
            manifest_sha256: manifest.fingerprint()?,
            tokenizer_sha256,
            timestamp_unix: unix_now(),
            scores,
            skipped,
        })
    }
}

/// One ordered score on a capability forgetting curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingPoint {
    /// Checkpoint or session identifier.
    pub checkpoint_or_session_id: String,
    /// Capability score.
    pub score: f64,
    /// Number of examples used.
    pub examples: usize,
    /// UNIX timestamp in seconds.
    pub timestamp_unix: u64,
    /// Underlying task scores.
    pub tasks: Vec<TaskScore>,
    /// Optional routing signatures.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routing: Vec<RoutingSignature>,
}

/// Ordered score history for one capability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingCurve {
    /// Capability represented by the curve.
    pub capability: Capability,
    /// Ordered checkpoint or session points.
    pub points: Vec<ForgettingPoint>,
}

impl ForgettingCurve {
    /// Return the point matching an identifier.
    pub fn point_at(&self, id: &str) -> Option<&ForgettingPoint> {
        self.points
            .iter()
            .find(|point| point.checkpoint_or_session_id == id)
    }

    /// Return the score matching an identifier.
    pub fn score_at(&self, id: &str) -> Option<f64> {
        self.point_at(id).map(|point| point.score)
    }

    /// Compare two points using an `after - before` signed delta.
    pub fn delta(
        &self,
        baseline_id: &str,
        current_id: &str,
        threshold: f64,
    ) -> Option<ForgettingDelta> {
        let before = self.score_at(baseline_id)?;
        let after = self.score_at(current_id)?;
        let delta = after - before;
        Some(ForgettingDelta {
            capability_or_concept: self.capability.to_string(),
            baseline_checkpoint_or_session: baseline_id.to_string(),
            current_checkpoint_or_session: current_id.to_string(),
            score_before: before,
            score_after: after,
            delta,
            significant: delta.abs() >= threshold,
        })
    }
}

/// Exact seven-field forgetting comparison shared with the optional Manas bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingDelta {
    /// Capability or external concept identifier.
    pub capability_or_concept: String,
    /// Baseline checkpoint or session identifier.
    pub baseline_checkpoint_or_session: String,
    /// Current checkpoint or session identifier.
    pub current_checkpoint_or_session: String,
    /// Baseline score.
    pub score_before: f64,
    /// Current score.
    pub score_after: f64,
    /// Signed `score_after - score_before` change.
    pub delta: f64,
    /// Whether the absolute change meets the configured threshold.
    pub significant: bool,
}

/// Alias with an explicit name for the future Manas JSONL bridge.
pub type ManasForgettingRecord = ForgettingDelta;

/// Persistent multi-capability forgetting history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingStore {
    /// Store schema version.
    pub schema_version: u32,
    /// Stable probe suite identifier.
    pub suite_id: String,
    /// Probe manifest SHA-256.
    pub manifest_sha256: String,
    /// Optional tokenizer SHA-256 used to prevent invalid comparisons.
    pub tokenizer_sha256: Option<String>,
    /// Significance threshold used by this series.
    pub significance_threshold: f64,
    /// Capability curves keyed for deterministic serialization.
    pub curves: BTreeMap<Capability, ForgettingCurve>,
    /// Skipped capabilities from the latest run.
    #[serde(default)]
    pub latest_skipped: Vec<ProbeSkip>,
}

impl ForgettingStore {
    /// Create an empty validated store.
    pub fn new(
        manifest: &ProbeManifest,
        tokenizer_sha256: Option<String>,
        significance_threshold: f64,
    ) -> Result<Self> {
        validate_threshold(significance_threshold)?;
        Ok(Self {
            schema_version: 1,
            suite_id: manifest.suite_id.clone(),
            manifest_sha256: manifest.fingerprint()?,
            tokenizer_sha256,
            significance_threshold,
            curves: BTreeMap::new(),
            latest_skipped: Vec::new(),
        })
    }

    /// Load a store from JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = fs::File::open(path)?;
        let store: Self = serde_json::from_reader(file)?;
        store.validate()?;
        Ok(store)
    }

    /// Load a store when present or create a new one.
    pub fn load_or_new(
        path: impl AsRef<Path>,
        manifest: &ProbeManifest,
        tokenizer_sha256: Option<String>,
        significance_threshold: f64,
    ) -> Result<Self> {
        if path.as_ref().exists() {
            let store = Self::load(path)?;
            store.validate_run_contract(
                manifest,
                tokenizer_sha256.as_deref(),
                significance_threshold,
            )?;
            Ok(store)
        } else {
            Self::new(manifest, tokenizer_sha256, significance_threshold)
        }
    }

    /// Validate the persisted store.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(AarambhError::Config(format!(
                "unsupported forgetting store schema {}, expected 1",
                self.schema_version
            )));
        }
        validate_id("forgetting suite_id", &self.suite_id)?;
        validate_threshold(self.significance_threshold)?;
        for (capability, curve) in &self.curves {
            if capability != &curve.capability {
                return Err(AarambhError::Config(format!(
                    "forgetting curve key '{capability}' does not match curve capability '{}'",
                    curve.capability
                )));
            }
            let mut ids = BTreeSet::new();
            for point in &curve.points {
                validate_point(point)?;
                if !ids.insert(&point.checkpoint_or_session_id) {
                    return Err(AarambhError::Config(format!(
                        "duplicate forgetting point '{}'",
                        point.checkpoint_or_session_id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Append one run, treating identical duplicate points as idempotent.
    pub fn record(&mut self, run: &ForgettingRun) -> Result<()> {
        self.validate_run(run)?;
        let points = run
            .scores
            .iter()
            .map(|score| {
                (
                    score.capability,
                    ForgettingPoint {
                        checkpoint_or_session_id: run.checkpoint_or_session_id.clone(),
                        score: score.score,
                        examples: score.examples,
                        timestamp_unix: run.timestamp_unix,
                        tasks: score.tasks.clone(),
                        routing: score.routing.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        for (capability, point) in &points {
            if let Some(existing) = self
                .curves
                .get(capability)
                .and_then(|curve| curve.point_at(&run.checkpoint_or_session_id))
                && !same_point(existing, point)
            {
                return Err(AarambhError::Config(format!(
                    "conflicting forgetting point '{}' for capability '{}'",
                    run.checkpoint_or_session_id, capability
                )));
            }
        }
        for (capability, point) in points {
            let curve = self
                .curves
                .entry(capability)
                .or_insert_with(|| ForgettingCurve {
                    capability,
                    points: Vec::new(),
                });
            if curve.point_at(&run.checkpoint_or_session_id).is_none() {
                curve.points.push(point);
            }
        }
        self.latest_skipped = run.skipped.clone();
        Ok(())
    }

    /// Return whether any capability curve contains the identifier.
    pub fn contains_checkpoint_or_session(&self, id: &str) -> bool {
        self.curves
            .values()
            .any(|curve| curve.point_at(id).is_some())
    }

    /// Return deltas for current capabilities after validating comparable task subsets.
    pub fn deltas(&self, baseline_id: &str, current_id: &str) -> Result<Vec<ForgettingDelta>> {
        let mut deltas = Vec::new();
        for curve in self.curves.values() {
            let Some(current) = curve.point_at(current_id) else {
                continue;
            };
            let baseline = curve.point_at(baseline_id).ok_or_else(|| {
                AarambhError::Config(format!(
                    "forgetting baseline '{baseline_id}' has no '{}' capability point",
                    curve.capability
                ))
            })?;
            if !same_eval_contract(baseline, current) {
                return Err(AarambhError::Config(format!(
                    "forgetting capability '{}' used different task subsets or example counts between '{baseline_id}' and '{current_id}'",
                    curve.capability
                )));
            }
            if let Some(delta) = curve.delta(baseline_id, current_id, self.significance_threshold) {
                deltas.push(delta);
            }
        }
        Ok(deltas)
    }

    /// Compute MoE routing drift where both points contain signatures.
    pub fn routing_drift(&self, baseline_id: &str, current_id: &str) -> Vec<RoutingDrift> {
        self.curves
            .values()
            .filter_map(|curve| {
                let before = curve.point_at(baseline_id)?;
                let after = curve.point_at(current_id)?;
                compare_routing(curve.capability, &before.routing, &after.routing)
            })
            .collect()
    }

    /// Save the complete store atomically as pretty JSON.
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let temporary = temporary_path(path);
        {
            let file = fs::File::create(&temporary)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, self)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        fs::rename(&temporary, path)?;
        Ok(())
    }

    /// Write exact seven-field JSONL records atomically.
    pub fn export_jsonl(
        &self,
        path: impl AsRef<Path>,
        baseline_id: &str,
        current_id: &str,
    ) -> Result<usize> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let records = self.deltas(baseline_id, current_id)?;
        let temporary = temporary_path(path);
        {
            let file = fs::File::create(&temporary)?;
            let mut writer = BufWriter::new(file);
            for record in &records {
                serde_json::to_writer(&mut writer, record)?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }
        fs::rename(&temporary, path)?;
        Ok(records.len())
    }

    /// Load and validate exact seven-field bridge records.
    pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<ManasForgettingRecord>> {
        let file = fs::File::open(path)?;
        let mut records = Vec::new();
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ManasForgettingRecord = serde_json::from_str(&line).map_err(|error| {
                AarambhError::Config(format!(
                    "invalid forgetting JSONL record on line {}: {error}",
                    index + 1
                ))
            })?;
            validate_delta(&record)?;
            records.push(record);
        }
        Ok(records)
    }

    fn validate_run_contract(
        &self,
        manifest: &ProbeManifest,
        tokenizer_sha256: Option<&str>,
        threshold: f64,
    ) -> Result<()> {
        if self.suite_id != manifest.suite_id || self.manifest_sha256 != manifest.fingerprint()? {
            return Err(AarambhError::Config(
                "forgetting store probe manifest does not match the current suite".into(),
            ));
        }
        if self.tokenizer_sha256.as_deref() != tokenizer_sha256 {
            return Err(AarambhError::Config(
                "forgetting store tokenizer fingerprint does not match".into(),
            ));
        }
        if (self.significance_threshold - threshold).abs() > f64::EPSILON {
            return Err(AarambhError::Config(format!(
                "forgetting store threshold {} does not match requested threshold {threshold}",
                self.significance_threshold
            )));
        }
        Ok(())
    }

    fn validate_run(&self, run: &ForgettingRun) -> Result<()> {
        if self.suite_id != run.suite_id || self.manifest_sha256 != run.manifest_sha256 {
            return Err(AarambhError::Config(
                "forgetting run probe suite does not match the store".into(),
            ));
        }
        if self.tokenizer_sha256 != run.tokenizer_sha256 {
            return Err(AarambhError::Config(
                "forgetting run tokenizer fingerprint does not match the store".into(),
            ));
        }
        validate_id("checkpoint_or_session_id", &run.checkpoint_or_session_id)?;
        validate_scores(&run.scores)
    }
}

/// Return one semantic SHA-256 for a tokenizer vocabulary and merge table.
pub fn tokenizer_fingerprint(tokenizer: &BpeTokenizer) -> Result<String> {
    let bytes = serde_json::to_vec(&(&tokenizer.vocab.id_to_token, &tokenizer.merges))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Execute all manifest probes while reusing the loaded model and tokenizer.
pub fn run_capability_probes(
    context: &EvalContext,
    config: &EvalConfig,
    manifest: &ProbeManifest,
    checkpoint_or_session_id: impl Into<String>,
    tokenizer_sha256: Option<String>,
    require_all: bool,
) -> Result<ForgettingRun> {
    manifest.validate()?;
    let mut scores = Vec::with_capacity(manifest.probes.len());
    let mut skipped = Vec::new();
    for probe in &manifest.probes {
        let mut probe_config = config.clone();
        probe_config.tasks = probe.tasks.clone();
        probe_config.max_examples = bounded_example_limit(probe.max_examples, config.max_examples);
        match run_all(context, &probe_config).and_then(|scorecard| {
            let mut score = aggregate_capability(probe.capability, scorecard.tasks)?;
            score.routing = collect_routing_signatures(context, &probe_config, probe)?;
            Ok(score)
        }) {
            Ok(score) => scores.push(score),
            Err(error) if !require_all && is_unavailable_probe_error(&error) => {
                skipped.push(ProbeSkip {
                    capability: probe.capability,
                    reason: error.to_string(),
                });
            }
            Err(error) => return Err(error),
        }
    }
    ForgettingRun::new(
        checkpoint_or_session_id,
        manifest,
        tokenizer_sha256,
        scores,
        skipped,
    )
}

fn is_unavailable_probe_error(error: &AarambhError) -> bool {
    match error {
        AarambhError::Io(error) => error.kind() == std::io::ErrorKind::NotFound,
        AarambhError::Unsupported(_) => true,
        AarambhError::Config(message) => {
            let message = message.to_ascii_lowercase();
            message.contains("requires --")
                || message.contains("requires [vision")
                || message.contains("no projector path configured")
        }
        _ => false,
    }
}

fn collect_routing_signatures(
    context: &EvalContext,
    config: &EvalConfig,
    probe: &CapabilityProbe,
) -> Result<Vec<RoutingSignature>> {
    if context.model().config().moe.is_none() {
        return Ok(Vec::new());
    }
    let mut signatures = Vec::new();
    for task in &probe.tasks {
        let normalized = normalize_task(task);
        let Some(path) = task_data_path(&config.data_dir, &normalized) else {
            continue;
        };
        let file = fs::File::open(&path)?;
        let limit = bounded_example_limit(probe.max_examples, config.max_examples);
        for (index, line) in BufReader::new(file).lines().enumerate() {
            if limit.is_some_and(|limit| index >= limit) {
                break;
            }
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)?;
            let prompt = routing_prompt(&value).ok_or_else(|| {
                AarambhError::Config(format!(
                    "forgetting routing probe could not extract text from {} line {}",
                    path.display(),
                    index + 1
                ))
            })?;
            let mut ids = context.tokenizer().encode(&prompt)?;
            if ids.is_empty() {
                continue;
            }
            if ids.len() > context.max_seq_len() {
                ids.truncate(context.max_seq_len());
            }
            let len = ids.len();
            let tensor = Tensor::from_vec(ids, (1, len), context.device())?;
            let experts_by_layer = context.model().routing_signature(&tensor)?;
            signatures.push(RoutingSignature {
                example_id: format!("{normalized}:{index}"),
                experts_by_layer,
            });
        }
    }
    Ok(signatures)
}

fn task_data_path(data_dir: &Path, task: &str) -> Option<PathBuf> {
    let candidates: &[&str] = match task {
        "mmlu" => &["mmlu_lite/data.jsonl"],
        "hellaswag" => &["hellaswag/data.jsonl"],
        "gsm8k" => &["gsm8k_subset/data.jsonl"],
        "humaneval" => &["humaneval_lite/data.jsonl"],
        "vqa" => &["vqa/data.jsonl", "vqa_smoke/data.jsonl"],
        "video-qa" => &["video_qa/data.jsonl", "video_qa_smoke/data.jsonl"],
        "document-qa" => &[
            "docvqa/data.jsonl",
            "document_qa/data.jsonl",
            "document_qa_smoke/data.jsonl",
        ],
        "tool-calling" => &["tool_calling/data.jsonl"],
        "tool-chain" => &["tool_chain/data.jsonl"],
        _ => return None,
    };
    candidates
        .iter()
        .map(|candidate| data_dir.join(candidate))
        .find(|path| path.exists())
}

fn routing_prompt(value: &serde_json::Value) -> Option<String> {
    let mut parts = Vec::new();
    for field in ["instruction", "question", "prompt", "context"] {
        if let Some(text) = value.get(field).and_then(serde_json::Value::as_str) {
            parts.push(text.to_string());
        }
    }
    for field in ["choices", "endings", "tools"] {
        if let Some(extra) = value.get(field) {
            parts.push(extra.to_string());
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn aggregate_capability(capability: Capability, tasks: Vec<TaskScore>) -> Result<CapabilityScore> {
    if tasks.is_empty() {
        return Err(AarambhError::Config(format!(
            "forgetting capability '{capability}' produced no task scores"
        )));
    }
    let mut weighted = 0.0;
    let mut examples = 0usize;
    for task in &tasks {
        if !task.higher_is_better || !task.value.is_finite() || !(0.0..=1.0).contains(&task.value) {
            return Err(AarambhError::Config(format!(
                "forgetting task '{}' must produce a finite higher-is-better score in [0, 1]",
                task.name
            )));
        }
        if task.examples == 0 {
            return Err(AarambhError::Config(format!(
                "forgetting task '{}' produced zero examples",
                task.name
            )));
        }
        weighted += task.value * task.examples as f64;
        examples = examples.saturating_add(task.examples);
    }
    Ok(CapabilityScore {
        capability,
        score: weighted / examples as f64,
        examples,
        tasks,
        routing: Vec::new(),
    })
}

fn compare_routing(
    capability: Capability,
    before: &[RoutingSignature],
    after: &[RoutingSignature],
) -> Option<RoutingDrift> {
    if before.is_empty() || after.is_empty() {
        return None;
    }
    let before = before
        .iter()
        .map(|signature| (signature.example_id.as_str(), signature))
        .collect::<BTreeMap<_, _>>();
    let after = after
        .iter()
        .map(|signature| (signature.example_id.as_str(), signature))
        .collect::<BTreeMap<_, _>>();
    let mut compared = 0usize;
    let mut changed = 0usize;
    for (id, baseline) in before {
        let Some(current) = after.get(id) else {
            continue;
        };
        compared += 1;
        changed += usize::from(baseline.experts_by_layer != current.experts_by_layer);
    }
    (compared > 0).then(|| RoutingDrift {
        capability,
        drift_rate: changed as f64 / compared as f64,
        compared_examples: compared,
        changed_examples: changed,
    })
}

fn validate_scores(scores: &[CapabilityScore]) -> Result<()> {
    let mut capabilities = BTreeSet::new();
    for score in scores {
        if !capabilities.insert(score.capability) {
            return Err(AarambhError::Config(format!(
                "duplicate capability score '{}'",
                score.capability
            )));
        }
        if !score.score.is_finite() || !(0.0..=1.0).contains(&score.score) {
            return Err(AarambhError::Config(format!(
                "capability '{}' score must be finite and in [0, 1]",
                score.capability
            )));
        }
        if score.examples == 0 {
            return Err(AarambhError::Config(format!(
                "capability '{}' must contain at least one example",
                score.capability
            )));
        }
    }
    Ok(())
}

fn validate_point(point: &ForgettingPoint) -> Result<()> {
    validate_id("checkpoint_or_session_id", &point.checkpoint_or_session_id)?;
    if !point.score.is_finite() || !(0.0..=1.0).contains(&point.score) {
        return Err(AarambhError::Config(
            "forgetting point score must be finite and in [0, 1]".into(),
        ));
    }
    if point.examples == 0 {
        return Err(AarambhError::Config(
            "forgetting point must contain at least one example".into(),
        ));
    }
    Ok(())
}

fn validate_delta(delta: &ForgettingDelta) -> Result<()> {
    validate_id("capability_or_concept", &delta.capability_or_concept)?;
    validate_id(
        "baseline_checkpoint_or_session",
        &delta.baseline_checkpoint_or_session,
    )?;
    validate_id(
        "current_checkpoint_or_session",
        &delta.current_checkpoint_or_session,
    )?;
    if !delta.score_before.is_finite()
        || !delta.score_after.is_finite()
        || !(0.0..=1.0).contains(&delta.score_before)
        || !(0.0..=1.0).contains(&delta.score_after)
        || !delta.delta.is_finite()
        || !(-1.0..=1.0).contains(&delta.delta)
    {
        return Err(AarambhError::Config(
            "forgetting JSONL scores must be in [0, 1] and delta in [-1, 1]".into(),
        ));
    }
    let expected = delta.score_after - delta.score_before;
    if (delta.delta - expected).abs() > 1e-9 {
        return Err(AarambhError::Config(
            "forgetting JSONL delta must equal score_after - score_before".into(),
        ));
    }
    Ok(())
}

fn validate_threshold(threshold: f64) -> Result<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(AarambhError::Config(
            "forgetting significance threshold must be finite and in [0, 1]".into(),
        ));
    }
    Ok(())
}

fn validate_id(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 256 || value.contains(['\n', '\r', '\0']) {
        return Err(AarambhError::Config(format!(
            "{name} must be non-empty, at most 256 bytes, and contain no control line breaks"
        )));
    }
    Ok(())
}

fn normalize_task(task: &str) -> String {
    match task.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "gsm8k-subset" => "gsm8k".into(),
        "humaneval-lite" => "humaneval".into(),
        "mmlu-lite" => "mmlu".into(),
        "vision-qa" | "vqa-smoke" => "vqa".into(),
        "nextqa" | "video-qa-smoke" => "video-qa".into(),
        "docvqa" | "document-qa-smoke" => "document-qa".into(),
        "function-calling" => "tool-calling".into(),
        "agent-chain" | "bfcl-multistep" => "tool-chain".into(),
        other => other.to_string(),
    }
}

fn same_point(left: &ForgettingPoint, right: &ForgettingPoint) -> bool {
    left.checkpoint_or_session_id == right.checkpoint_or_session_id
        && (left.score - right.score).abs() <= f64::EPSILON
        && left.examples == right.examples
        && left.tasks == right.tasks
        && left.routing == right.routing
}

fn same_eval_contract(left: &ForgettingPoint, right: &ForgettingPoint) -> bool {
    left.examples == right.examples
        && left.tasks.len() == right.tasks.len()
        && left.tasks.iter().zip(&right.tasks).all(|(left, right)| {
            left.name == right.name
                && left.metric == right.metric
                && left.higher_is_better == right.higher_is_better
                && left.examples == right.examples
        })
}

fn bounded_example_limit(probe: Option<usize>, run: Option<usize>) -> Option<usize> {
    match (probe, run) {
        (Some(probe), Some(run)) => Some(probe.min(run)),
        (Some(probe), None) => Some(probe),
        (None, Some(run)) => Some(run),
        (None, None) => None,
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("forgetting.json");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    path.with_file_name(format!(".{file_name}.{}.{nonce}.tmp", std::process::id()))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> ProbeManifest {
        ProbeManifest {
            schema_version: 1,
            suite_id: "phase38-smoke".into(),
            probes: vec![
                CapabilityProbe {
                    capability: Capability::Math,
                    tasks: vec!["gsm8k".into()],
                    max_examples: Some(2),
                },
                CapabilityProbe {
                    capability: Capability::ToolUse,
                    tasks: vec!["tool-calling".into(), "tool-chain".into()],
                    max_examples: Some(2),
                },
            ],
        }
    }

    fn run(id: &str, math: f64) -> ForgettingRun {
        ForgettingRun::new(
            id,
            &manifest(),
            Some("tokenizer".into()),
            vec![CapabilityScore {
                capability: Capability::Math,
                score: math,
                examples: 2,
                tasks: vec![TaskScore::accuracy(
                    "gsm8k",
                    (math * 2.0).round() as usize,
                    2,
                )],
                routing: Vec::new(),
            }],
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn curve_tracks_multiple_points_and_signed_delta() {
        let manifest = manifest();
        let mut store = ForgettingStore::new(&manifest, Some("tokenizer".into()), 0.02).unwrap();
        store.record(&run("base", 0.75)).unwrap();
        store.record(&run("middle", 0.74)).unwrap();
        store.record(&run("current", 0.70)).unwrap();
        let delta = &store.deltas("base", "current").unwrap()[0];
        assert!((delta.delta + 0.05).abs() < 1e-9);
        assert!(delta.significant);
        assert_eq!(store.curves[&Capability::Math].points.len(), 3);
    }

    #[test]
    fn threshold_treats_small_delta_as_noise() {
        let manifest = manifest();
        let mut store = ForgettingStore::new(&manifest, Some("tokenizer".into()), 0.02).unwrap();
        store.record(&run("base", 0.75)).unwrap();
        store.record(&run("current", 0.731)).unwrap();
        assert!(!store.deltas("base", "current").unwrap()[0].significant);
    }

    #[test]
    fn comparison_rejects_different_example_counts() {
        let manifest = manifest();
        let mut store = ForgettingStore::new(&manifest, Some("tokenizer".into()), 0.02).unwrap();
        store.record(&run("base", 0.75)).unwrap();
        let current = ForgettingRun::new(
            "current",
            &manifest,
            Some("tokenizer".into()),
            vec![CapabilityScore {
                capability: Capability::Math,
                score: 1.0,
                examples: 1,
                tasks: vec![TaskScore::accuracy("gsm8k", 1, 1)],
                routing: Vec::new(),
            }],
            Vec::new(),
        )
        .unwrap();
        store.record(&current).unwrap();
        assert!(store.deltas("base", "current").is_err());
    }

    #[test]
    fn identical_duplicate_is_idempotent_but_conflict_fails() {
        let manifest = manifest();
        let mut store = ForgettingStore::new(&manifest, Some("tokenizer".into()), 0.02).unwrap();
        store.record(&run("base", 0.75)).unwrap();
        store.record(&run("base", 0.75)).unwrap();
        assert_eq!(store.curves[&Capability::Math].points.len(), 1);
        assert!(store.record(&run("base", 0.5)).is_err());
    }

    #[test]
    fn conflicting_run_is_rejected_transactionally() {
        let manifest = manifest();
        let mut store = ForgettingStore::new(&manifest, Some("tokenizer".into()), 0.02).unwrap();
        store.record(&run("base", 0.75)).unwrap();
        let conflicting = ForgettingRun::new(
            "base",
            &manifest,
            Some("tokenizer".into()),
            vec![
                CapabilityScore {
                    capability: Capability::ToolUse,
                    score: 0.5,
                    examples: 2,
                    tasks: vec![TaskScore::accuracy("tool-calling", 1, 2)],
                    routing: Vec::new(),
                },
                CapabilityScore {
                    capability: Capability::Math,
                    score: 0.5,
                    examples: 2,
                    tasks: vec![TaskScore::accuracy("gsm8k", 1, 2)],
                    routing: Vec::new(),
                },
            ],
            Vec::new(),
        )
        .unwrap();
        assert!(store.record(&conflicting).is_err());
        assert!(!store.curves.contains_key(&Capability::ToolUse));
    }

    #[test]
    fn manifest_rejects_task_owned_by_another_capability() {
        let mut invalid = manifest();
        invalid.probes[0].tasks = vec!["mmlu".into()];
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn routing_drift_compares_matching_examples() {
        let before = vec![RoutingSignature {
            example_id: "one".into(),
            experts_by_layer: vec![vec![0, 2]],
        }];
        let after = vec![RoutingSignature {
            example_id: "one".into(),
            experts_by_layer: vec![vec![1, 2]],
        }];
        let drift = compare_routing(Capability::Math, &before, &after).unwrap();
        assert_eq!(drift.changed_examples, 1);
        assert_eq!(drift.drift_rate, 1.0);
    }

    #[test]
    fn bridge_json_has_exact_seven_fields() {
        let record = ForgettingDelta {
            capability_or_concept: "math".into(),
            baseline_checkpoint_or_session: "before".into(),
            current_checkpoint_or_session: "after".into(),
            score_before: 0.8,
            score_after: 0.7,
            delta: -0.1,
            significant: true,
        };
        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 7);
    }

    #[test]
    fn bridge_validation_rejects_scores_outside_schema_bounds() {
        let record = ForgettingDelta {
            capability_or_concept: "math".into(),
            baseline_checkpoint_or_session: "before".into(),
            current_checkpoint_or_session: "after".into(),
            score_before: 1.1,
            score_after: 0.7,
            delta: -0.4,
            significant: true,
        };
        assert!(validate_delta(&record).is_err());
    }

    #[test]
    fn only_expected_availability_errors_can_be_skipped() {
        assert!(is_unavailable_probe_error(&AarambhError::Io(
            std::io::Error::from(std::io::ErrorKind::NotFound)
        )));
        assert!(is_unavailable_probe_error(&AarambhError::Config(
            "VQA eval requires [vision] config".into()
        )));
        assert!(!is_unavailable_probe_error(&AarambhError::Shape(
            "bad tensor".into()
        )));
        assert!(!is_unavailable_probe_error(&AarambhError::Json(
            serde_json::from_str::<serde_json::Value>("{").unwrap_err()
        )));
    }

    #[test]
    fn run_level_example_limit_can_only_reduce_probe_cost() {
        assert_eq!(bounded_example_limit(Some(16), Some(2)), Some(2));
        assert_eq!(bounded_example_limit(Some(4), Some(8)), Some(4));
        assert_eq!(bounded_example_limit(None, Some(8)), Some(8));
    }
}
