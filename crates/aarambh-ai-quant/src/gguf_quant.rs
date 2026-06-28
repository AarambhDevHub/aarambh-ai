pub const Q4_K_M_BLOCK_SIZE: usize = 256;
pub const Q4_K_M_ENCODED_SIZE: usize = 132;

pub fn quantise_block_q4_k_m(block_256_weights: &[f32]) -> [u8; Q4_K_M_ENCODED_SIZE] {
    assert_eq!(block_256_weights.len(), Q4_K_M_BLOCK_SIZE);
    let min = block_256_weights
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max = block_256_weights
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let scale = if (max - min).abs() <= f32::EPSILON {
        1.0
    } else {
        (max - min) / 15.0
    };

    let mut encoded = [0u8; Q4_K_M_ENCODED_SIZE];
    encoded[0..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
    encoded[2..4].copy_from_slice(&f32_to_f16(min).to_le_bytes());

    for pair_idx in 0..(Q4_K_M_BLOCK_SIZE / 2) {
        let a = block_256_weights[pair_idx * 2];
        let b = block_256_weights[pair_idx * 2 + 1];
        let qa = ((a - min) / scale).round().clamp(0.0, 15.0) as u8;
        let qb = ((b - min) / scale).round().clamp(0.0, 15.0) as u8;
        encoded[4 + pair_idx] = qa | (qb << 4);
    }

    encoded
}

pub fn dequantise_block_q4_k_m(block: &[u8; Q4_K_M_ENCODED_SIZE]) -> [f32; Q4_K_M_BLOCK_SIZE] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let mut values = [0.0f32; Q4_K_M_BLOCK_SIZE];
    for pair_idx in 0..(Q4_K_M_BLOCK_SIZE / 2) {
        let byte = block[4 + pair_idx];
        values[pair_idx * 2] = ((byte & 0x0f) as f32) * scale + min;
        values[pair_idx * 2 + 1] = ((byte >> 4) as f32) * scale + min;
    }
    values
}

pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 255 {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mant | 0x800000;
        let shift = 14 - half_exp;
        let mut half_mant = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x1000) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exp = ((value >> 10) & 0x1f) as i32;
    let mant = (value & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut mant_norm = mant;
            let mut exp_norm = -14;
            while (mant_norm & 0x0400) == 0 {
                mant_norm <<= 1;
                exp_norm -= 1;
            }
            mant_norm &= 0x03ff;
            sign | (((exp_norm + 127) as u32) << 23) | (mant_norm << 13)
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_is_reasonable() {
        for value in [-12.5f32, -1.0, 0.0, 0.5, 3.25, 100.0] {
            let decoded = f16_to_f32(f32_to_f16(value));
            assert!((decoded - value).abs() < 0.05, "{value} -> {decoded}");
        }
    }

    #[test]
    fn gguf_q4_block_roundtrip() {
        let weights: [f32; Q4_K_M_BLOCK_SIZE] =
            std::array::from_fn(|idx| (idx as f32) * 0.01 - 1.28);
        let block = quantise_block_q4_k_m(&weights);
        let dequant = dequantise_block_q4_k_m(&block);
        let max_err = weights
            .iter()
            .zip(dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 0.09, "max_err={max_err}");
    }
}
