/// Parallel fixed-state Gated DeltaNet recurrence.
pub mod gated_delta;
/// Parallel CPU attention kernels.
pub mod parallel_attn;
/// CPU RMSNorm kernels with runtime SIMD selection.
pub mod simd_norm;
