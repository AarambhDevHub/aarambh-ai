use std::path::Path;

use aarambh_ai_core::{AarambhError, Result};
use candle_core::{Device, Tensor};
use image::{ImageReader, Rgb, RgbImage, imageops::FilterType};
use serde::{Deserialize, Serialize};

/// Configuration for CLIP-style image preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionPreprocessConfig {
    /// Square image size after resize and center crop.
    pub image_size: usize,
    /// Per-channel RGB mean.
    pub mean: [f32; 3],
    /// Per-channel RGB standard deviation.
    pub std: [f32; 3],
}

impl Default for VisionPreprocessConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
        }
    }
}

/// Image preprocessor matching CLIP RGB normalization.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    config: VisionPreprocessConfig,
}

impl ImagePreprocessor {
    /// Create a preprocessor from explicit configuration.
    pub fn new(config: VisionPreprocessConfig) -> Result<Self> {
        if config.image_size == 0 {
            return Err(AarambhError::Config(
                "vision preprocess image_size must be non-zero".into(),
            ));
        }
        if config.std.iter().any(|value| *value <= 0.0) {
            return Err(AarambhError::Config(
                "vision preprocess std values must be positive".into(),
            ));
        }
        Ok(Self { config })
    }

    /// Return the preprocessing configuration.
    pub fn config(&self) -> &VisionPreprocessConfig {
        &self.config
    }

    /// Decode an image path and return normalized `[3, image_size, image_size]` pixels.
    pub fn preprocess_path(&self, path: impl AsRef<Path>, device: &Device) -> Result<Tensor> {
        let reader = ImageReader::open(path.as_ref()).map_err(|err| {
            AarambhError::Io(std::io::Error::new(
                err.kind(),
                format!("failed to open image {}: {err}", path.as_ref().display()),
            ))
        })?;
        let image = reader.decode().map_err(|err| {
            AarambhError::Config(format!(
                "failed to decode image {}: {err}",
                path.as_ref().display()
            ))
        })?;
        self.preprocess_rgb(&image.to_rgb8(), device)
    }

    /// Preprocess an already decoded RGB image.
    pub fn preprocess_rgb(&self, image: &RgbImage, device: &Device) -> Result<Tensor> {
        self.preprocess_rgb_values(image, device)
    }

    /// Preprocess RGB frames on CPU, concatenate them, and transfer one contiguous batch.
    pub fn preprocess_rgb_batch(&self, images: &[RgbImage], device: &Device) -> Result<Tensor> {
        self.preprocess_rgb_batch_with(images, device, |preprocessor, image, cpu| {
            preprocessor.preprocess_rgb_values(image, cpu)
        })
    }

    /// Preprocess document pages with aspect-preserving resize and white padding.
    pub fn preprocess_document_pages(
        &self,
        images: &[RgbImage],
        device: &Device,
    ) -> Result<Tensor> {
        self.preprocess_rgb_batch_with(images, device, |preprocessor, image, cpu| {
            let fitted = preprocessor.fit_and_pad_rgb(image)?;
            preprocessor.normalize_rgb(&fitted, cpu)
        })
    }

    fn preprocess_rgb_batch_with(
        &self,
        images: &[RgbImage],
        device: &Device,
        mut preprocess: impl FnMut(&Self, &RgbImage, &Device) -> Result<Tensor>,
    ) -> Result<Tensor> {
        if images.is_empty() {
            return Err(AarambhError::Config(
                "batched vision preprocessing requires at least one image".into(),
            ));
        }
        let cpu = Device::Cpu;
        let frames = images
            .iter()
            .map(|image| preprocess(self, image, &cpu))
            .collect::<Result<Vec<_>>>()?;
        let references = frames.iter().collect::<Vec<_>>();
        Ok(Tensor::stack(&references, 0)?
            .contiguous()?
            .to_device(device)?)
    }

    fn preprocess_rgb_values(&self, image: &RgbImage, device: &Device) -> Result<Tensor> {
        let image_size = self.config.image_size as u32;
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err(AarambhError::Config(
                "image dimensions must be non-zero".into(),
            ));
        }

        let scale = image_size as f32 / width.min(height) as f32;
        let resized_width = ((width as f32 * scale).round() as u32).max(image_size);
        let resized_height = ((height as f32 * scale).round() as u32).max(image_size);
        let resized =
            image::imageops::resize(image, resized_width, resized_height, FilterType::CatmullRom);
        let left = (resized_width - image_size) / 2;
        let top = (resized_height - image_size) / 2;
        let cropped =
            image::imageops::crop_imm(&resized, left, top, image_size, image_size).to_image();
        self.normalize_rgb(&cropped, device)
    }

    fn fit_and_pad_rgb(&self, image: &RgbImage) -> Result<RgbImage> {
        let image_size = self.config.image_size as u32;
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err(AarambhError::Config(
                "image dimensions must be non-zero".into(),
            ));
        }
        let scale = image_size as f32 / width.max(height) as f32;
        let resized_width = ((width as f32 * scale).round() as u32).clamp(1, image_size);
        let resized_height = ((height as f32 * scale).round() as u32).clamp(1, image_size);
        let resized =
            image::imageops::resize(image, resized_width, resized_height, FilterType::CatmullRom);
        let mut canvas = RgbImage::from_pixel(image_size, image_size, Rgb([255, 255, 255]));
        let left = (image_size - resized_width) / 2;
        let top = (image_size - resized_height) / 2;
        image::imageops::replace(&mut canvas, &resized, i64::from(left), i64::from(top));
        Ok(canvas)
    }

    fn normalize_rgb(&self, image: &RgbImage, device: &Device) -> Result<Tensor> {
        let size = self.config.image_size;
        if image.dimensions() != (size as u32, size as u32) {
            return Err(AarambhError::Shape(format!(
                "normalized image must be {size}x{size}, got {}x{}",
                image.width(),
                image.height()
            )));
        }
        let plane = size * size;
        let mut values = vec![0f32; 3 * plane];
        for y in 0..size {
            for x in 0..size {
                let pixel = image.get_pixel(x as u32, y as u32);
                let idx = y * size + x;
                for channel in 0..3 {
                    let value = pixel[channel] as f32 / 255.0;
                    values[channel * plane + idx] =
                        (value - self.config.mean[channel]) / self.config.std[channel];
                }
            }
        }

        Ok(Tensor::from_vec(values, (3, size, size), device)?)
    }
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self::new(VisionPreprocessConfig::default()).expect("default vision preprocess config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn preprocess_rgb_returns_clip_tensor_shape() {
        let image = ImageBuffer::from_fn(320, 240, |x, y| {
            Rgb([(x % 255) as u8, (y % 255) as u8, 128])
        });
        let pre = ImagePreprocessor::default();
        let tensor = pre.preprocess_rgb(&image, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[3, 224, 224]);
    }

    #[test]
    fn preprocess_rgb_batch_returns_contiguous_nchw() {
        let images = vec![
            ImageBuffer::from_pixel(32, 32, Rgb([255, 0, 0])),
            ImageBuffer::from_pixel(32, 32, Rgb([0, 255, 0])),
        ];
        let pre = ImagePreprocessor::new(VisionPreprocessConfig {
            image_size: 16,
            ..VisionPreprocessConfig::default()
        })
        .unwrap();
        let tensor = pre.preprocess_rgb_batch(&images, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[2, 3, 16, 16]);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn document_preprocess_preserves_orientation_with_padding() {
        let images = vec![
            ImageBuffer::from_pixel(32, 16, Rgb([255, 0, 0])),
            ImageBuffer::from_pixel(16, 32, Rgb([0, 255, 0])),
        ];
        let pre = ImagePreprocessor::new(VisionPreprocessConfig {
            image_size: 16,
            ..VisionPreprocessConfig::default()
        })
        .unwrap();
        let tensor = pre
            .preprocess_document_pages(&images, &Device::Cpu)
            .unwrap();
        assert_eq!(tensor.dims(), &[2, 3, 16, 16]);
        assert!(tensor.is_contiguous());
    }
}
