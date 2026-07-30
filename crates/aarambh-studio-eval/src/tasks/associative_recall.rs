use aarambh_studio_core::Result;
use serde::Deserialize;

use crate::generation::greedy_generate;
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct Association {
    key: String,
    value: String,
}

#[derive(Debug, Clone, Deserialize)]
struct AssociativeRecallExample {
    pairs: Vec<Association>,
    query: String,
    answer: String,
}

/// Exact-match key-value retrieval task for recurrent-memory models.
pub struct AssociativeRecallTask;

impl EvalTask for AssociativeRecallTask {
    fn name(&self) -> &'static str {
        "associative-recall"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config
            .data_dir
            .join("associative_recall")
            .join("data.jsonl");
        let examples = read_jsonl::<AssociativeRecallExample>(&path, config.max_examples)?;
        let mut correct = 0usize;

        for example in &examples {
            let prompt = render_prompt(example);
            let completion = greedy_generate(context, &prompt, config.max_new_tokens)?;
            if answer_matches(&completion, &example.answer) {
                correct += 1;
            }
        }

        Ok(TaskScore::accuracy(
            "associative-recall",
            correct,
            examples.len(),
        ))
    }
}

fn render_prompt(example: &AssociativeRecallExample) -> String {
    let mut prompt = String::from("Remember these key-value pairs.\n");
    for pair in &example.pairs {
        prompt.push_str(&pair.key);
        prompt.push_str(": ");
        prompt.push_str(&pair.value);
        prompt.push('\n');
    }
    prompt.push_str("Value for ");
    prompt.push_str(&example.query);
    prompt.push(':');
    prompt
}

fn answer_matches(completion: &str, answer: &str) -> bool {
    let expected = normalize(answer);
    completion
        .split_whitespace()
        .next()
        .is_some_and(|token| normalize(token) == expected)
}

fn normalize(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorer_accepts_only_the_first_exact_value() {
        assert!(answer_matches("cobalt\n", "cobalt"));
        assert!(answer_matches("Cobalt.", "cobalt"));
        assert!(!answer_matches("dark cobalt", "cobalt"));
        assert!(!answer_matches("cobalt-blue", "cobalt"));
    }

    #[test]
    fn prompt_preserves_pair_order_and_query() {
        let example = AssociativeRecallExample {
            pairs: vec![Association {
                key: "K7".into(),
                value: "cobalt".into(),
            }],
            query: "K7".into(),
            answer: "cobalt".into(),
        };
        assert_eq!(
            render_prompt(&example),
            "Remember these key-value pairs.\nK7: cobalt\nValue for K7:"
        );
    }
}
