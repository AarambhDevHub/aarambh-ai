use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct CpuGatedDelta {
    batch: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

/// Run one FP32 Gated DeltaNet recurrent update on CPU.
pub fn cpu_gated_delta_recurrent(
    packed: &Tensor,
    state: &Tensor,
    key_dim: usize,
    value_dim: usize,
) -> Result<Tensor> {
    let (batch, heads, width) = packed.dims3()?;
    let expected_width = key_dim * 2 + value_dim + 2;
    if width != expected_width || state.dims() != [batch, heads, key_dim, value_dim] {
        return Err(Error::msg(format!(
            "invalid gated delta shapes packed={:?}, state={:?}, key_dim={key_dim}, value_dim={value_dim}",
            packed.dims(),
            state.dims()
        )));
    }
    packed.apply_op2_no_bwd(
        state,
        &CpuGatedDelta {
            batch,
            heads,
            key_dim,
            value_dim,
        },
    )
}

impl CustomOp2 for CpuGatedDelta {
    fn name(&self) -> &'static str {
        "aarambh-gated-delta-cpu"
    }

    fn cpu_fwd(
        &self,
        packed_storage: &CpuStorage,
        packed_layout: &Layout,
        state_storage: &CpuStorage,
        state_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let packed = f32_slice(packed_storage, packed_layout, self.name())?;
        let state = f32_slice(state_storage, state_layout, self.name())?;
        let rows = self.batch * self.heads;
        let packed_width = self.key_dim * 2 + self.value_dim + 2;
        let state_width = self.key_dim * self.value_dim;
        let output_width = state_width + self.value_dim;
        let mut output = vec![0f32; rows * output_width];

        packed
            .par_chunks_exact(packed_width)
            .zip(state.par_chunks_exact(state_width))
            .zip(output.par_chunks_exact_mut(output_width))
            .for_each(|((input, previous), result)| {
                update_row(input, previous, result, self.key_dim, self.value_dim)
            });
        Ok((
            CpuStorage::F32(output),
            Shape::from((self.batch, self.heads, output_width)),
        ))
    }
}

fn update_row(input: &[f32], previous: &[f32], output: &mut [f32], dk: usize, dv: usize) {
    let q = &input[..dk];
    let k = &input[dk..2 * dk];
    let v = &input[2 * dk..2 * dk + dv];
    let alpha = input[2 * dk + dv];
    let beta = input[2 * dk + dv + 1];
    let (next, values) = output.split_at_mut(dk * dv);

    for value_idx in 0..dv {
        let mut prediction = 0.0f32;
        for key_idx in 0..dk {
            prediction += k[key_idx] * previous[key_idx * dv + value_idx] * alpha;
        }
        let error = v[value_idx] - prediction;
        let mut mixed = 0.0f32;
        for key_idx in 0..dk {
            let index = key_idx * dv + value_idx;
            let updated = previous[index] * alpha + beta * k[key_idx] * error;
            next[index] = updated;
            mixed += q[key_idx] * updated;
        }
        values[value_idx] = mixed;
    }
}

fn f32_slice<'a>(storage: &'a CpuStorage, layout: &Layout, op: &'static str) -> Result<&'a [f32]> {
    let values = match storage {
        CpuStorage::F32(values) => values,
        other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), op).bt()),
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| Error::RequiresContiguous { op }.bt())?;
    Ok(&values[start..end])
}
