use std::collections::BTreeSet;

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_tokenizer::{BpeTokenizer, tool_json_token_text};
use serde_json::{Map, Value};

const MAX_SCHEMA_DEPTH: usize = 32;
const MAX_SCHEMA_NODES: usize = 4096;
#[derive(Debug, Clone)]
/// Compiled practical JSON Schema subset used by function-call decoding.
pub struct JsonSchema {
    root: SchemaNode,
}

impl JsonSchema {
    /// Compile a JSON Schema value and reject unsupported or malformed keywords.
    pub fn compile(value: &Value) -> Result<Self> {
        let mut nodes = 0usize;
        let root = SchemaNode::compile(value, "$", 0, &mut nodes)?;
        Ok(Self { root })
    }

    /// Return true when the schema root is an object.
    pub fn is_object(&self) -> bool {
        self.root.is_object()
    }

    /// Validate a complete JSON value against the compiled schema.
    pub fn validate(&self, value: &Value) -> Result<()> {
        self.root.validate(value, "$")
    }
}

#[derive(Debug, Clone)]
/// Incremental token-level JSON grammar constrained by a compiled schema.
pub struct JsonSchemaGrammar {
    alternatives: Vec<SchemaNode>,
    text: String,
    complete: bool,
}

impl JsonSchemaGrammar {
    /// Create an empty grammar for one schema.
    pub fn new(schema: JsonSchema) -> Self {
        Self::from_nodes(vec![schema.root])
    }

    pub(crate) fn from_nodes(alternatives: Vec<SchemaNode>) -> Self {
        Self {
            alternatives,
            text: String::new(),
            complete: false,
        }
    }

    /// Return the generated JSON prefix.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return true when the prefix is one complete schema-valid JSON value.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Compute token ids that keep the current JSON prefix viable.
    pub fn allowed_token_ids(&self, tokenizer: &BpeTokenizer) -> Result<Vec<u32>> {
        if self.complete {
            return Ok(Vec::new());
        }
        let mut allowed = Vec::new();
        let mut candidate = String::with_capacity(self.text.len() + 32);
        for (token_id, piece) in tokenizer.vocab.id_to_token.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            let piece = tool_json_token_text(token_id as u32, tokenizer)?;
            candidate.clear();
            candidate.push_str(&self.text);
            candidate.push_str(&piece);
            if self.status(&candidate) != PrefixStatus::Invalid {
                allowed.push(token_id as u32);
            }
        }
        if allowed.is_empty() {
            return Err(AarambhError::Config(format!(
                "JSON grammar has no valid next token after {:?}",
                self.text
            )));
        }
        Ok(allowed)
    }

    /// Commit one decoded token fragment to the grammar.
    pub fn accept_token(&mut self, token_text: &str) -> Result<()> {
        self.text.push_str(token_text);
        match self.status(&self.text) {
            PrefixStatus::Invalid => Err(AarambhError::Config(format!(
                "token produced invalid JSON grammar prefix {:?}",
                self.text
            ))),
            PrefixStatus::Incomplete => Ok(()),
            PrefixStatus::Complete => {
                self.complete = true;
                Ok(())
            }
        }
    }

    pub(crate) fn accept_token_id(
        &mut self,
        token_id: u32,
        tokenizer: &BpeTokenizer,
    ) -> Result<()> {
        let text = tool_json_token_text(token_id, tokenizer)?;
        self.accept_token(&text)
    }

    /// Parse and validate the completed JSON value.
    pub fn finish(&self) -> Result<Value> {
        if !self.complete {
            return Err(AarambhError::Config(
                "JSON grammar is incomplete at generation end".into(),
            ));
        }
        let value: Value = serde_json::from_str(&self.text)?;
        if self
            .alternatives
            .iter()
            .any(|schema| schema.validate(&value, "$").is_ok())
        {
            Ok(value)
        } else {
            Err(AarambhError::Config(
                "completed JSON does not match any compiled schema".into(),
            ))
        }
    }

    fn status(&self, text: &str) -> PrefixStatus {
        let bytes = text.as_bytes();
        let mut incomplete = false;
        for schema in &self.alternatives {
            let outcome = parse_node(schema, bytes, 0);
            if outcome
                .completions
                .iter()
                .any(|completion| completion.pos == bytes.len())
            {
                return PrefixStatus::Complete;
            }
            incomplete |= outcome.incomplete;
        }
        if incomplete {
            PrefixStatus::Incomplete
        } else {
            PrefixStatus::Invalid
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixStatus {
    Invalid,
    Incomplete,
    Complete,
}

#[derive(Debug, Clone)]
pub(crate) enum SchemaNode {
    Object(ObjectSchema),
    Array(ArraySchema),
    String(StringSchema),
    Integer(NumberSchema),
    Number(NumberSchema),
    Boolean,
    Null,
    Literal(Vec<Value>),
    Union(Vec<SchemaNode>),
}

#[derive(Debug, Clone)]
pub(crate) struct ObjectSchema {
    properties: Vec<(String, SchemaNode)>,
    required: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ArraySchema {
    items: Box<SchemaNode>,
    min_items: usize,
    max_items: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct StringSchema {
    min_len: usize,
    max_len: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct NumberSchema {
    minimum: Option<(f64, bool)>,
    maximum: Option<(f64, bool)>,
}

impl SchemaNode {
    fn compile(value: &Value, path: &str, depth: usize, nodes: &mut usize) -> Result<Self> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(AarambhError::Config(format!(
                "JSON Schema exceeds maximum depth {MAX_SCHEMA_DEPTH} at {path}"
            )));
        }
        *nodes += 1;
        if *nodes > MAX_SCHEMA_NODES {
            return Err(AarambhError::Config(format!(
                "JSON Schema exceeds maximum node count {MAX_SCHEMA_NODES}"
            )));
        }
        let object = value.as_object().ok_or_else(|| {
            AarambhError::Config(format!("JSON Schema at {path} must be an object"))
        })?;
        reject_unsupported_keywords(object, path)?;

        if let Some(constant) = object.get("const") {
            validate_scalar_literal(constant, &format!("{path}.const"))?;
            return Ok(Self::Literal(vec![constant.clone()]));
        }
        if let Some(values) = object.get("enum") {
            let values = values
                .as_array()
                .ok_or_else(|| AarambhError::Config(format!("{path}.enum must be an array")))?;
            if values.is_empty() {
                return Err(AarambhError::Config(format!(
                    "{path}.enum must not be empty"
                )));
            }
            for (index, value) in values.iter().enumerate() {
                validate_scalar_literal(value, &format!("{path}.enum[{index}]"))?;
            }
            return Ok(Self::Literal(values.clone()));
        }

        let types = schema_types(object, path)?;
        if types.len() > 1 {
            if types.len() != 2 || !types.iter().any(|kind| kind == "null") {
                return Err(AarambhError::Unsupported(format!(
                    "{path}.type only supports one type or one type plus null"
                )));
            }
            let mut variants = Vec::with_capacity(2);
            for kind in types {
                variants.push(Self::compile_type(&kind, object, path, depth, nodes)?);
            }
            return Ok(Self::Union(variants));
        }
        Self::compile_type(&types[0], object, path, depth, nodes)
    }

    fn compile_type(
        kind: &str,
        object: &Map<String, Value>,
        path: &str,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<Self> {
        match kind {
            "object" => {
                let empty = Map::new();
                let properties_object = match object.get("properties") {
                    Some(value) => value.as_object().ok_or_else(|| {
                        AarambhError::Config(format!("{path}.properties must be an object"))
                    })?,
                    None => &empty,
                };
                let properties = properties_object
                    .iter()
                    .map(|(name, schema)| {
                        Ok((
                            name.clone(),
                            Self::compile(
                                schema,
                                &format!("{path}.properties.{name}"),
                                depth + 1,
                                nodes,
                            )?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let required = object
                    .get("required")
                    .map(|value| parse_string_set(value, &format!("{path}.required")))
                    .transpose()?
                    .unwrap_or_default();
                let names = properties
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<BTreeSet<_>>();
                if let Some(missing) = required.iter().find(|name| !names.contains(name.as_str())) {
                    return Err(AarambhError::Config(format!(
                        "{path}.required contains unknown property {missing:?}"
                    )));
                }
                if object
                    .get("additionalProperties")
                    .is_some_and(|value| !value.is_boolean())
                {
                    return Err(AarambhError::Unsupported(format!(
                        "{path}.additionalProperties must be boolean in the function subset"
                    )));
                }
                Ok(Self::Object(ObjectSchema {
                    properties,
                    required,
                }))
            }
            "array" => {
                let items = object.get("items").ok_or_else(|| {
                    AarambhError::Config(format!("{path}.items is required for arrays"))
                })?;
                let min_items = usize_keyword(object, "minItems", 0, path)?;
                let max_items = optional_usize_keyword(object, "maxItems", path)?.or(Some(16));
                if max_items.is_some_and(|max| max < min_items) {
                    return Err(AarambhError::Config(format!(
                        "{path}.maxItems must be at least minItems"
                    )));
                }
                Ok(Self::Array(ArraySchema {
                    items: Box::new(Self::compile(
                        items,
                        &format!("{path}.items"),
                        depth + 1,
                        nodes,
                    )?),
                    min_items,
                    max_items,
                }))
            }
            "string" => {
                let min_len = usize_keyword(object, "minLength", 0, path)?;
                let max_len = optional_usize_keyword(object, "maxLength", path)?.or(Some(64));
                if max_len.is_some_and(|max| max < min_len) {
                    return Err(AarambhError::Config(format!(
                        "{path}.maxLength must be at least minLength"
                    )));
                }
                Ok(Self::String(StringSchema { min_len, max_len }))
            }
            "integer" => Ok(Self::Integer(number_schema(object, path)?)),
            "number" => Ok(Self::Number(number_schema(object, path)?)),
            "boolean" => Ok(Self::Boolean),
            "null" => Ok(Self::Null),
            other => Err(AarambhError::Unsupported(format!(
                "unsupported JSON Schema type {other:?} at {path}"
            ))),
        }
    }

    fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    fn validate(&self, value: &Value, path: &str) -> Result<()> {
        match self {
            Self::Object(schema) => {
                let object = value
                    .as_object()
                    .ok_or_else(|| type_error(path, "object"))?;
                for required in &schema.required {
                    if !object.contains_key(required) {
                        return Err(AarambhError::Config(format!(
                            "{path} is missing required property {required:?}"
                        )));
                    }
                }
                for (name, child) in &schema.properties {
                    if let Some(value) = object.get(name) {
                        child.validate(value, &format!("{path}.{name}"))?;
                    }
                }
                if let Some(name) = object
                    .keys()
                    .find(|name| !schema.properties.iter().any(|(known, _)| known == *name))
                {
                    return Err(AarambhError::Config(format!(
                        "{path} contains unsupported property {name:?}"
                    )));
                }
                Ok(())
            }
            Self::Array(schema) => {
                let array = value.as_array().ok_or_else(|| type_error(path, "array"))?;
                if array.len() < schema.min_items
                    || schema.max_items.is_some_and(|max| array.len() > max)
                {
                    return Err(AarambhError::Config(format!(
                        "{path} array length is outside schema bounds"
                    )));
                }
                for (index, value) in array.iter().enumerate() {
                    schema.items.validate(value, &format!("{path}[{index}]"))?;
                }
                Ok(())
            }
            Self::String(schema) => {
                let text = value.as_str().ok_or_else(|| type_error(path, "string"))?;
                let len = text.chars().count();
                if len < schema.min_len || schema.max_len.is_some_and(|max| len > max) {
                    return Err(AarambhError::Config(format!(
                        "{path} string length is outside schema bounds"
                    )));
                }
                Ok(())
            }
            Self::Integer(schema) => {
                let number = value
                    .as_i64()
                    .map(|value| value as f64)
                    .or_else(|| value.as_u64().map(|value| value as f64))
                    .ok_or_else(|| type_error(path, "integer"))?;
                validate_number(schema, number, path)
            }
            Self::Number(schema) => {
                let number = value.as_f64().ok_or_else(|| type_error(path, "number"))?;
                validate_number(schema, number, path)
            }
            Self::Boolean => value
                .is_boolean()
                .then_some(())
                .ok_or_else(|| type_error(path, "boolean")),
            Self::Null => value
                .is_null()
                .then_some(())
                .ok_or_else(|| type_error(path, "null")),
            Self::Literal(values) => values
                .iter()
                .any(|expected| expected == value)
                .then_some(())
                .ok_or_else(|| AarambhError::Config(format!("{path} is not an allowed literal"))),
            Self::Union(variants) => variants
                .iter()
                .any(|schema| schema.validate(value, path).is_ok())
                .then_some(())
                .ok_or_else(|| AarambhError::Config(format!("{path} matches no allowed type"))),
        }
    }
}

fn reject_unsupported_keywords(object: &Map<String, Value>, path: &str) -> Result<()> {
    const UNSUPPORTED: &[&str] = &[
        "$ref",
        "$defs",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "pattern",
        "patternProperties",
        "dependentRequired",
        "dependentSchemas",
        "prefixItems",
        "contains",
        "unevaluatedProperties",
        "multipleOf",
    ];
    if let Some(keyword) = UNSUPPORTED
        .iter()
        .find(|keyword| object.contains_key(**keyword))
    {
        return Err(AarambhError::Unsupported(format!(
            "JSON Schema keyword {keyword:?} is not supported at {path}"
        )));
    }
    Ok(())
}

fn schema_types(object: &Map<String, Value>, path: &str) -> Result<Vec<String>> {
    let inferred = if object.contains_key("properties") {
        "object"
    } else if object.contains_key("items") {
        "array"
    } else {
        "string"
    };
    match object.get("type") {
        None => Ok(vec![inferred.into()]),
        Some(Value::String(kind)) => Ok(vec![kind.clone()]),
        Some(Value::Array(types)) => {
            let mut out = Vec::with_capacity(types.len());
            for value in types {
                out.push(
                    value
                        .as_str()
                        .ok_or_else(|| {
                            AarambhError::Config(format!(
                                "{path}.type array values must be strings"
                            ))
                        })?
                        .to_string(),
                );
            }
            if out.is_empty() {
                return Err(AarambhError::Config(format!(
                    "{path}.type must not be empty"
                )));
            }
            Ok(out)
        }
        Some(_) => Err(AarambhError::Config(format!(
            "{path}.type must be a string or string array"
        ))),
    }
}

fn parse_string_set(value: &Value, path: &str) -> Result<BTreeSet<String>> {
    value
        .as_array()
        .ok_or_else(|| AarambhError::Config(format!("{path} must be an array")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| AarambhError::Config(format!("{path} values must be strings")))
        })
        .collect()
}

fn usize_keyword(
    object: &Map<String, Value>,
    key: &str,
    default: usize,
    path: &str,
) -> Result<usize> {
    optional_usize_keyword(object, key, path).map(|value| value.unwrap_or(default))
}

fn optional_usize_keyword(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<usize>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    AarambhError::Config(format!("{path}.{key} must be a non-negative integer"))
                })
        })
        .transpose()
}

fn number_schema(object: &Map<String, Value>, path: &str) -> Result<NumberSchema> {
    let minimum = number_bound(object, "minimum", false, path)?.or(number_bound(
        object,
        "exclusiveMinimum",
        true,
        path,
    )?);
    let maximum = number_bound(object, "maximum", false, path)?.or(number_bound(
        object,
        "exclusiveMaximum",
        true,
        path,
    )?);
    if minimum.is_some_and(|(min, _)| maximum.is_some_and(|(max, _)| min > max)) {
        return Err(AarambhError::Config(format!(
            "{path} minimum exceeds maximum"
        )));
    }
    Ok(NumberSchema { minimum, maximum })
}

fn number_bound(
    object: &Map<String, Value>,
    key: &str,
    exclusive: bool,
    path: &str,
) -> Result<Option<(f64, bool)>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| (value, exclusive))
                .ok_or_else(|| AarambhError::Config(format!("{path}.{key} must be finite")))
        })
        .transpose()
}

fn validate_scalar_literal(value: &Value, path: &str) -> Result<()> {
    if value.is_null() || value.is_boolean() || value.is_number() || value.is_string() {
        Ok(())
    } else {
        Err(AarambhError::Unsupported(format!(
            "{path} only supports scalar JSON values"
        )))
    }
}

fn validate_number(schema: &NumberSchema, value: f64, path: &str) -> Result<()> {
    if schema.minimum.is_some_and(|(bound, exclusive)| {
        if exclusive {
            value <= bound
        } else {
            value < bound
        }
    }) || schema.maximum.is_some_and(|(bound, exclusive)| {
        if exclusive {
            value >= bound
        } else {
            value > bound
        }
    }) {
        return Err(AarambhError::Config(format!(
            "{path} number is outside schema bounds"
        )));
    }
    Ok(())
}

fn type_error(path: &str, expected: &str) -> AarambhError {
    AarambhError::Config(format!("{path} must be {expected}"))
}

#[derive(Default)]
struct ParseOutcome {
    completions: Vec<Completion>,
    incomplete: bool,
}

struct Completion {
    pos: usize,
    value: Value,
}

impl ParseOutcome {
    fn incomplete() -> Self {
        Self {
            completions: Vec::new(),
            incomplete: true,
        }
    }

    fn complete(pos: usize, value: Value) -> Self {
        Self {
            completions: vec![Completion { pos, value }],
            incomplete: false,
        }
    }

    fn merge(&mut self, other: Self) {
        self.completions.extend(other.completions);
        self.incomplete |= other.incomplete;
    }
}

fn parse_node(schema: &SchemaNode, input: &[u8], pos: usize) -> ParseOutcome {
    match schema {
        SchemaNode::Object(schema) => parse_object(schema, input, pos),
        SchemaNode::Array(schema) => parse_array(schema, input, pos),
        SchemaNode::String(schema) => parse_string(schema, input, pos),
        SchemaNode::Integer(schema) => parse_number(schema, true, input, pos),
        SchemaNode::Number(schema) => parse_number(schema, false, input, pos),
        SchemaNode::Boolean => parse_literals(&[Value::Bool(true), Value::Bool(false)], input, pos),
        SchemaNode::Null => parse_literals(&[Value::Null], input, pos),
        SchemaNode::Literal(values) => parse_literals(values, input, pos),
        SchemaNode::Union(variants) => {
            let mut outcome = ParseOutcome::default();
            for variant in variants {
                outcome.merge(parse_node(variant, input, pos));
            }
            outcome
        }
    }
}

fn parse_object(schema: &ObjectSchema, input: &[u8], pos: usize) -> ParseOutcome {
    let Some(pos) = consume_byte(input, pos, b'{') else {
        return prefix_byte(input, pos, b'{');
    };
    let mut states = vec![(pos, Map::new(), 0usize)];
    let mut outcome = ParseOutcome::default();
    for (name, child) in &schema.properties {
        let mut next_states = Vec::new();
        let key = serde_json::to_string(name).expect("property name serialization");
        for (state_pos, object, emitted) in states {
            if !schema.required.contains(name) {
                next_states.push((state_pos, object.clone(), emitted));
            }
            let prefix = if emitted == 0 {
                format!("{key}:")
            } else {
                format!(",{key}:")
            };
            match consume_literal(input, state_pos, prefix.as_bytes()) {
                LiteralMatch::Incomplete => outcome.incomplete = true,
                LiteralMatch::Invalid => {}
                LiteralMatch::Complete(value_pos) => {
                    let parsed = parse_node(child, input, value_pos);
                    outcome.incomplete |= parsed.incomplete;
                    for completion in parsed.completions {
                        let mut object = object.clone();
                        object.insert(name.clone(), completion.value);
                        next_states.push((completion.pos, object, emitted + 1));
                    }
                }
            }
        }
        states = next_states;
        if states.is_empty() {
            return outcome;
        }
    }
    for (state_pos, object, _) in states {
        match consume_literal(input, state_pos, b"}") {
            LiteralMatch::Incomplete => outcome.incomplete = true,
            LiteralMatch::Invalid => {}
            LiteralMatch::Complete(pos) => {
                outcome.completions.push(Completion {
                    pos,
                    value: Value::Object(object),
                });
            }
        }
    }
    outcome
}

fn parse_array(schema: &ArraySchema, input: &[u8], pos: usize) -> ParseOutcome {
    let Some(pos) = consume_byte(input, pos, b'[') else {
        return prefix_byte(input, pos, b'[');
    };
    parse_array_items(schema, input, pos, Vec::new())
}

fn parse_array_items(
    schema: &ArraySchema,
    input: &[u8],
    pos: usize,
    values: Vec<Value>,
) -> ParseOutcome {
    if pos == input.len() {
        return ParseOutcome::incomplete();
    }
    let mut outcome = ParseOutcome::default();
    if values.len() >= schema.min_items && input[pos] == b']' {
        outcome.completions.push(Completion {
            pos: pos + 1,
            value: Value::Array(values.clone()),
        });
    }
    if schema.max_items.is_some_and(|max| values.len() >= max) {
        return outcome;
    }
    let item_pos = if values.is_empty() {
        pos
    } else if input[pos] == b',' {
        pos + 1
    } else {
        return outcome;
    };
    let parsed = parse_node(&schema.items, input, item_pos);
    outcome.incomplete |= parsed.incomplete;
    for completion in parsed.completions {
        let mut next = values.clone();
        next.push(completion.value);
        outcome.merge(parse_array_items(schema, input, completion.pos, next));
    }
    outcome
}

fn parse_string(schema: &StringSchema, input: &[u8], pos: usize) -> ParseOutcome {
    if pos >= input.len() {
        return ParseOutcome::incomplete();
    }
    if input[pos] != b'"' {
        return ParseOutcome::default();
    }
    let mut index = pos + 1;
    let mut escaped = false;
    while index < input.len() {
        let byte = input[index];
        if escaped {
            if byte == b'u' {
                if input.len() < index + 5 {
                    return if input[index + 1..].iter().all(u8::is_ascii_hexdigit) {
                        ParseOutcome::incomplete()
                    } else {
                        ParseOutcome::default()
                    };
                }
                if !input[index + 1..index + 5]
                    .iter()
                    .all(u8::is_ascii_hexdigit)
                {
                    return ParseOutcome::default();
                }
                index += 5;
                escaped = false;
                continue;
            }
            if !matches!(byte, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                return ParseOutcome::default();
            }
            escaped = false;
            index += 1;
            continue;
        }
        match byte {
            b'\\' => {
                escaped = true;
                index += 1;
            }
            b'"' => {
                let slice = &input[pos..=index];
                let Ok(Value::String(text)) = serde_json::from_slice::<Value>(slice) else {
                    return ParseOutcome::default();
                };
                let len = text.chars().count();
                if len < schema.min_len || schema.max_len.is_some_and(|max| len > max) {
                    return ParseOutcome::default();
                }
                return ParseOutcome::complete(index + 1, Value::String(text));
            }
            0x00..=0x1f => return ParseOutcome::default(),
            _ => index += 1,
        }
    }
    if schema.max_len.is_some_and(|max| {
        std::str::from_utf8(&input[pos + 1..])
            .map(|text| text.chars().count() > max)
            .unwrap_or(false)
    }) {
        ParseOutcome::default()
    } else {
        ParseOutcome::incomplete()
    }
}

fn parse_number(schema: &NumberSchema, integer: bool, input: &[u8], pos: usize) -> ParseOutcome {
    if pos >= input.len() {
        return ParseOutcome::incomplete();
    }
    let mut end = pos;
    while end < input.len() && matches!(input[end], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
    {
        end += 1;
    }
    if end == pos {
        return ParseOutcome::default();
    }
    let text = match std::str::from_utf8(&input[pos..end]) {
        Ok(text) => text,
        Err(_) => return ParseOutcome::default(),
    };
    let parsed = serde_json::from_str::<Value>(text).ok();
    let value = parsed.and_then(|value| {
        if integer && !(value.as_i64().is_some() || value.as_u64().is_some()) {
            return None;
        }
        let number = value.as_f64()?;
        validate_number(schema, number, "$").ok()?;
        Some(value)
    });
    let can_extend = end == input.len() && text.len() < 64 && number_prefix_viable(text, integer);
    let mut outcome = ParseOutcome {
        completions: Vec::new(),
        incomplete: can_extend,
    };
    if let Some(value) = value {
        outcome.completions.push(Completion { pos: end, value });
    }
    outcome
}

fn number_prefix_viable(text: &str, integer: bool) -> bool {
    if text.is_empty() || text == "-" {
        return true;
    }
    if integer {
        return text
            .strip_prefix('-')
            .unwrap_or(text)
            .chars()
            .all(|char| char.is_ascii_digit());
    }
    text.chars()
        .all(|char| char.is_ascii_digit() || matches!(char, '-' | '+' | '.' | 'e' | 'E'))
}

fn parse_literals(values: &[Value], input: &[u8], pos: usize) -> ParseOutcome {
    let mut outcome = ParseOutcome::default();
    for value in values {
        let literal = serde_json::to_vec(value).expect("literal serialization");
        match consume_literal(input, pos, &literal) {
            LiteralMatch::Incomplete => outcome.incomplete = true,
            LiteralMatch::Invalid => {}
            LiteralMatch::Complete(pos) => outcome.completions.push(Completion {
                pos,
                value: value.clone(),
            }),
        }
    }
    outcome
}

enum LiteralMatch {
    Invalid,
    Incomplete,
    Complete(usize),
}

fn consume_literal(input: &[u8], pos: usize, literal: &[u8]) -> LiteralMatch {
    if pos > input.len() {
        return LiteralMatch::Invalid;
    }
    let remaining = &input[pos..];
    let common = remaining.len().min(literal.len());
    if remaining[..common] != literal[..common] {
        return LiteralMatch::Invalid;
    }
    if remaining.len() < literal.len() {
        LiteralMatch::Incomplete
    } else {
        LiteralMatch::Complete(pos + literal.len())
    }
}

fn consume_byte(input: &[u8], pos: usize, expected: u8) -> Option<usize> {
    (pos < input.len() && input[pos] == expected).then_some(pos + 1)
}

fn prefix_byte(input: &[u8], pos: usize, expected: u8) -> ParseOutcome {
    if pos == input.len() {
        ParseOutcome::incomplete()
    } else if input[pos] == expected {
        ParseOutcome::complete(pos + 1, Value::Null)
    } else {
        ParseOutcome::default()
    }
}

pub(crate) fn tool_call_schema(name: &str, arguments: &JsonSchema) -> SchemaNode {
    SchemaNode::Object(ObjectSchema {
        properties: vec![
            (
                "name".into(),
                SchemaNode::Literal(vec![Value::String(name.into())]),
            ),
            ("arguments".into(), arguments.root.clone()),
        ],
        required: BTreeSet::from(["arguments".into(), "name".into()]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_nested_function_schema() {
        let schema = JsonSchema::compile(&serde_json::json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "items": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
            },
            "required": ["count"]
        }))
        .unwrap();
        schema
            .validate(&serde_json::json!({"count": 2, "items": ["a"]}))
            .unwrap();
        assert!(schema.validate(&serde_json::json!({"count": 0})).is_err());
    }

    #[test]
    fn grammar_accepts_valid_prefix_and_completion() {
        let schema = JsonSchema::compile(&serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }))
        .unwrap();
        let mut grammar = JsonSchemaGrammar::new(schema);
        grammar.accept_token(r#"{"city":"Del"#).unwrap();
        assert!(!grammar.is_complete());
        grammar.accept_token(r#"hi"}"#).unwrap();
        assert!(grammar.is_complete());
        assert_eq!(grammar.finish().unwrap()["city"], "Delhi");
    }

    #[test]
    fn unsupported_ref_is_rejected() {
        let error = JsonSchema::compile(&serde_json::json!({"$ref": "#/$defs/x"})).unwrap_err();
        assert!(error.to_string().contains("$ref"));
    }

    #[test]
    fn optional_properties_may_be_omitted_or_emitted_in_schema_order() {
        let schema = JsonSchema::compile(&serde_json::json!({
            "type":"object",
            "properties":{
                "city":{"type":"string"},
                "unit":{"type":"string","enum":["celsius","fahrenheit"]}
            },
            "required":["city"]
        }))
        .unwrap();
        let mut omitted = JsonSchemaGrammar::new(schema.clone());
        omitted.accept_token(r#"{"city":"Delhi"}"#).unwrap();
        assert!(omitted.is_complete());

        let mut emitted = JsonSchemaGrammar::new(schema);
        emitted
            .accept_token(r#"{"city":"Delhi","unit":"celsius"}"#)
            .unwrap();
        assert!(emitted.is_complete());
    }

    #[test]
    fn array_and_string_bounds_are_enforced() {
        let schema = JsonSchema::compile(&serde_json::json!({
            "type":"array",
            "items":{"type":"string","minLength":2,"maxLength":3},
            "minItems":1,
            "maxItems":2
        }))
        .unwrap();
        schema.validate(&serde_json::json!(["ab", "xyz"])).unwrap();
        assert!(schema.validate(&serde_json::json!([])).is_err());
        assert!(schema.validate(&serde_json::json!(["a"])).is_err());
        assert!(
            schema
                .validate(&serde_json::json!(["ab", "cd", "ef"]))
                .is_err()
        );
    }
}
