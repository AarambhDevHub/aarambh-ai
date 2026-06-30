/// Prompt-injection detector.
pub mod injection;
/// Jailbreak detector.
pub mod jailbreak;
/// PII detector and redactor.
pub mod pii;

pub use injection::{InjectionScore, detect_injection};
pub use jailbreak::{JailbreakScore, detect_jailbreak};
pub use pii::{PiiFinding, PiiFindings, PiiKind, detect_pii, redact_pii};

pub(crate) fn normalize_signal(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut previous_space = false;
    for ch in text.chars() {
        if is_zero_width(ch) {
            continue;
        }
        let mapped = map_confusable(ch).to_ascii_lowercase();
        if mapped.is_ascii_whitespace() {
            if !previous_space {
                normalized.push(' ');
                previous_space = true;
            }
        } else {
            normalized.push(mapped);
            previous_space = false;
        }
    }
    normalized
}

fn is_zero_width(ch: char) -> bool {
    matches!(
        ch,
        '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{feff}'
    )
}

fn map_confusable(ch: char) -> char {
    match ch {
        '0' | 'о' | 'ο' | 'Ο' | 'О' => 'o',
        '1' | '!' | 'í' | 'ì' | 'ï' | 'İ' => 'i',
        '3' | 'е' | 'Ε' | 'Е' => 'e',
        '4' | '@' | 'а' | 'Α' | 'А' => 'a',
        '5' | '$' => 's',
        '7' => 't',
        'р' | 'Ρ' | 'Р' => 'p',
        'с' | 'ϲ' | 'С' => 'c',
        'х' | 'Χ' | 'Х' => 'x',
        _ => ch,
    }
}
