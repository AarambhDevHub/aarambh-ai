use aarambh_ai_core::TokenizerLike;
use aarambh_ai_inference::{GenerationPhase, GenerationStep};
use aarambh_ai_tokenizer::BpeTokenizer;

pub fn render(
    step: &GenerationStep,
    tokenizer: &BpeTokenizer,
    temperature: f32,
    top_p: f32,
) -> String {
    let mut out = String::new();
    out.push_str("\nNext token predictions:\n");
    out.push_str("══════════════════════════════════════════════════════\n");
    out.push_str(&format!(
        "Phase: {} | Token: {}{}\n",
        phase_label(step.phase),
        step.token_id,
        if step.forced { " | forced" } else { "" }
    ));

    let mut selected_seen = false;
    for candidate in &step.candidates {
        let text = tokenizer
            .decode(&[candidate.token_id])
            .unwrap_or_else(|_| format!("<{}>", candidate.token_id));
        let pct = candidate.probability * 100.0;
        let bars = ((pct / 2.0).round() as usize).clamp(1, 24);
        let marker = if candidate.token_id == step.token_id {
            selected_seen = true;
            if step.forced { "  forced" } else { "  chosen" }
        } else {
            ""
        };
        out.push_str(&format!(
            "{:<24} {:>6.2}%  {:?}{}\n",
            "█".repeat(bars),
            pct,
            printable(&text),
            marker
        ));
    }

    if step.forced && !selected_seen {
        out.push_str(&format!(
            "{:<24} {:>6}  {:?}  forced\n",
            "",
            "",
            printable(&step.token_text)
        ));
    }

    out.push_str("══════════════════════════════════════════════════════\n");
    out.push_str(&format!(
        "Temperature: {:.2} | Top-P: {:.2} | Step: {}\n",
        temperature, top_p, step.step
    ));
    out
}

fn printable(text: &str) -> String {
    text.replace('\n', "\\n").replace('\t', "\\t")
}

fn phase_label(phase: GenerationPhase) -> &'static str {
    match phase {
        GenerationPhase::Thinking => "thinking",
        GenerationPhase::Answer => "answer",
    }
}
