//! Bounded, caller-executed, long-horizon tool-use chains.
#![deny(missing_docs)]

/// Multi-turn tool-chain orchestration.
pub mod chain;
/// Interactive and replay-based tool-result ingestion.
pub mod result_ingestion;
/// Exact-token chain state and result protocol types.
pub mod state;

pub use chain::{
    AgentError, AgentResult, ChainDecoder, ChainEvent, ChainMetrics, ChainOutput, ToolChain,
    ToolChainConfig,
};
pub use result_ingestion::{
    ReplayEntry, ReplayResultProvider, StdinResultProvider, ToolResultProvider,
};
pub use state::{
    ChainState, EvictionPolicy, ToolExchange, ToolResult, ToolResultContent, ToolResultRequest,
    ToolResultStatus,
};
