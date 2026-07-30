use aarambh_studio_core::{AarambhError, Result};
use candle_core::Tensor;

/// Replace one video token and its frame separators with projected frame patch tokens.
///
/// `frame_embeddings` must be `[frames, patches, hidden]`. Each frame after the first
/// is inserted immediately after the corresponding frame-separator text embedding.
pub fn interleave_video_tokens(
    text_tokens: &[u32],
    text_embeddings: &Tensor,
    frame_embeddings: &Tensor,
    video_placeholder_id: u32,
    frame_separator_id: u32,
) -> Result<Tensor> {
    interleave_media_sequence_tokens(
        text_tokens,
        text_embeddings,
        frame_embeddings,
        video_placeholder_id,
        frame_separator_id,
        "video",
    )
}

/// Replace one document token and its page separators with projected page patch tokens.
///
/// `page_embeddings` must be `[pages, patches, hidden]`. Each page after the first
/// is inserted immediately after its corresponding page-separator text embedding.
pub fn interleave_document_tokens(
    text_tokens: &[u32],
    text_embeddings: &Tensor,
    page_embeddings: &Tensor,
    document_placeholder_id: u32,
    page_separator_id: u32,
) -> Result<Tensor> {
    interleave_media_sequence_tokens(
        text_tokens,
        text_embeddings,
        page_embeddings,
        document_placeholder_id,
        page_separator_id,
        "document",
    )
}

/// Interleave an ordered visual sequence at one placeholder and separator positions.
pub fn interleave_media_sequence_tokens(
    text_tokens: &[u32],
    text_embeddings: &Tensor,
    media_embeddings: &Tensor,
    placeholder_id: u32,
    separator_id: u32,
    media_name: &str,
) -> Result<Tensor> {
    let text_dims = text_embeddings.dims();
    let media_dims = media_embeddings.dims();
    if text_dims.len() != 3 || text_dims[0] != 1 || text_dims[1] != text_tokens.len() {
        return Err(AarambhError::Shape(format!(
            "text_embeddings must be [1, {}, hidden], got {text_dims:?}",
            text_tokens.len()
        )));
    }
    if media_dims.len() != 3 || media_dims[0] == 0 {
        return Err(AarambhError::Shape(format!(
            "{media_name}_embeddings must be [items, patches, hidden], got {media_dims:?}"
        )));
    }
    if text_dims[2] != media_dims[2] {
        return Err(AarambhError::Shape(format!(
            "text hidden dim {} does not match {media_name} hidden dim {}",
            text_dims[2], media_dims[2]
        )));
    }
    let placeholders = text_tokens
        .iter()
        .filter(|token| **token == placeholder_id)
        .count();
    let separators = text_tokens
        .iter()
        .filter(|token| **token == separator_id)
        .count();
    if placeholders != 1 || separators != media_dims[0] - 1 {
        return Err(AarambhError::Config(format!(
            "{media_name} prompt requires one placeholder and {} separators; found {placeholders} and {separators}",
            media_dims[0] - 1
        )));
    }

    let mut next_item = 0usize;
    let mut parts = Vec::with_capacity(text_tokens.len() + media_dims[0]);
    for (index, token) in text_tokens.iter().enumerate() {
        if *token == placeholder_id {
            parts.push(media_embeddings.narrow(0, 0, 1)?);
            next_item = 1;
            continue;
        }
        parts.push(text_embeddings.narrow(1, index, 1)?);
        if *token == separator_id {
            parts.push(media_embeddings.narrow(0, next_item, 1)?);
            next_item += 1;
        }
    }
    Ok(Tensor::cat(&parts, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleave_image_tokens;
    use candle_core::Device;

    #[test]
    fn interleave_preserves_separator_embeddings() {
        let text = Tensor::from_vec(
            (0..8).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let frames =
            Tensor::from_vec(vec![100f32, 101., 200., 201.], (2, 1, 2), &Device::Cpu).unwrap();
        let fused = interleave_video_tokens(&[9, 11, 10, 20], &text, &frames, 9, 11).unwrap();
        assert_eq!(
            fused.squeeze(0).unwrap().to_vec2::<f32>().unwrap(),
            vec![
                vec![100., 101.],
                vec![2., 3.],
                vec![200., 201.],
                vec![4., 5.],
                vec![6., 7.]
            ]
        );
    }

    #[test]
    fn one_frame_matches_image_fusion() {
        let text =
            Tensor::from_vec(vec![7f32, 8., 20., 21., 30., 31.], (1, 3, 2), &Device::Cpu).unwrap();
        let patches =
            Tensor::from_vec(vec![100f32, 101., 110., 111.], (1, 2, 2), &Device::Cpu).unwrap();
        let image = interleave_image_tokens(&[7, 8, 20], &text, &patches, 7).unwrap();
        let video = interleave_video_tokens(&[9, 10, 20], &text, &patches, 9, 11).unwrap();
        assert_eq!(
            image.to_vec3::<f32>().unwrap(),
            video.to_vec3::<f32>().unwrap()
        );
    }
}
