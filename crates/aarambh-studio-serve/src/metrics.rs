use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use serde::Serialize;

#[derive(Debug, Default)]
/// Lock-free counters shared by HTTP handlers and the inference worker.
pub struct ServerMetrics {
    queued: AtomicUsize,
    active: AtomicUsize,
    requests_total: AtomicU64,
    requests_completed: AtomicU64,
    requests_cancelled: AtomicU64,
    requests_rejected: AtomicU64,
    safety_blocked: AtomicU64,
    generated_tokens: AtomicU64,
    decode_batches: AtomicU64,
    batch_items: AtomicU64,
    inference_errors: AtomicU64,
}

impl ServerMetrics {
    pub(crate) fn request_queued(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_queue_rollback(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        self.requests_total.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn request_admitted(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        self.active.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_completed(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_cancelled(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.requests_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_rejected(&self) {
        self.requests_rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn safety_blocked(&self) {
        self.safety_blocked.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn generated_token(&self) {
        self.generated_tokens.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn decode_batch(&self, size: usize) {
        self.decode_batches.fetch_add(1, Ordering::Relaxed);
        self.batch_items.fetch_add(size as u64, Ordering::Relaxed);
    }

    pub(crate) fn inference_error(&self) {
        self.inference_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Capture all counters as a serializable value.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let decode_batches = self.decode_batches.load(Ordering::Relaxed);
        let batch_items = self.batch_items.load(Ordering::Relaxed);
        MetricsSnapshot {
            queued: self.queued.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_cancelled: self.requests_cancelled.load(Ordering::Relaxed),
            requests_rejected: self.requests_rejected.load(Ordering::Relaxed),
            safety_blocked: self.safety_blocked.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            decode_batches,
            average_batch_size: if decode_batches == 0 {
                0.0
            } else {
                batch_items as f64 / decode_batches as f64
            },
            inference_errors: self.inference_errors.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
/// Point-in-time inference server metrics.
pub struct MetricsSnapshot {
    /// Requests waiting for admission.
    pub queued: usize,
    /// Requests currently generating.
    pub active: usize,
    /// Requests accepted since startup.
    pub requests_total: u64,
    /// Requests completed successfully.
    pub requests_completed: u64,
    /// Requests cancelled after client disconnect.
    pub requests_cancelled: u64,
    /// Requests rejected because the queue was full.
    pub requests_rejected: u64,
    /// Requests stopped by output safety.
    pub safety_blocked: u64,
    /// Generated token count.
    pub generated_tokens: u64,
    /// Shared decode pass count.
    pub decode_batches: u64,
    /// Mean number of requests per shared decode pass.
    pub average_batch_size: f64,
    /// Inference failures.
    pub inference_errors: u64,
}
