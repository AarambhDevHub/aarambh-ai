use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use aarambh_ai_core::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
}

impl Vocab {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        Ok(serde_json::to_writer(writer, self)?)
    }

    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }
}
