use aarambh_studio_core::{AarambhError, Result};
use candle_core::Tensor;

/// Replace one image placeholder token embedding with projected image tokens.
pub fn interleave_image_tokens(
    text_tokens: &[u32],
    text_embeddings: &Tensor,
    image_embeddings: &Tensor,
    image_placeholder_id: u32,
) -> Result<Tensor> {
    let text_dims = text_embeddings.dims();
    let image_dims = image_embeddings.dims();
    if text_dims.len() != 3 || text_dims[0] != 1 {
        return Err(AarambhError::Shape(format!(
            "text_embeddings must have shape [1, seq, hidden_dim], got {text_dims:?}"
        )));
    }
    if image_dims.len() != 3 || image_dims[0] != 1 {
        return Err(AarambhError::Shape(format!(
            "image_embeddings must have shape [1, image_tokens, hidden_dim], got {image_dims:?}"
        )));
    }
    if text_dims[1] != text_tokens.len() {
        return Err(AarambhError::Shape(format!(
            "text token count {} does not match text embedding seq {}",
            text_tokens.len(),
            text_dims[1]
        )));
    }
    if text_dims[2] != image_dims[2] {
        return Err(AarambhError::Shape(format!(
            "text hidden dim {} does not match image hidden dim {}",
            text_dims[2], image_dims[2]
        )));
    }

    let placeholder_count = text_tokens
        .iter()
        .filter(|token| **token == image_placeholder_id)
        .count();
    if placeholder_count != 1 {
        return Err(AarambhError::Config(format!(
            "expected exactly one image placeholder token id {image_placeholder_id}, found {placeholder_count}"
        )));
    }

    let mut parts = Vec::with_capacity(text_tokens.len() + 1);
    for (idx, token_id) in text_tokens.iter().enumerate() {
        if *token_id == image_placeholder_id {
            parts.push(image_embeddings.clone());
        } else {
            parts.push(text_embeddings.narrow(1, idx, 1)?);
        }
    }
    Ok(Tensor::cat(&parts, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn image_token_interleave_preserves_text_token_order() {
        let device = Device::Cpu;
        let text_tokens = vec![10u32, 7, 11, 12];
        let text_values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let image_values = vec![100f32, 101., 102., 103., 104., 105.];
        let text = Tensor::from_vec(text_values, (1, 4, 3), &device).unwrap();
        let image = Tensor::from_vec(image_values, (1, 2, 3), &device).unwrap();
        let fused = interleave_image_tokens(&text_tokens, &text, &image, 7).unwrap();
        assert_eq!(fused.dims(), &[1, 5, 3]);
        let rows = fused.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(rows[0], vec![0., 1., 2.]);
        assert_eq!(rows[1], vec![100., 101., 102.]);
        assert_eq!(rows[2], vec![103., 104., 105.]);
        assert_eq!(rows[3], vec![6., 7., 8.]);
        assert_eq!(rows[4], vec![9., 10., 11.]);
    }
}
