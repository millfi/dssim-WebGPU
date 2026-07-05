struct U32Buf {
    values: array<u32>,
};

struct F32Buf {
    values: array<f32>,
};

struct Vec4Buf {
    values: array<vec4<f32>>,
};

struct Params {
    len: u32,
    width: u32,
    height: u32,
    unused: u32,
};

@group(0) @binding(0) var<storage, read> in_rgba8: U32Buf;
@group(0) @binding(1) var<storage, read_write> out_linear: Vec4Buf;
@group(0) @binding(2) var<storage, read> srgb_to_linear_lut: F32Buf;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let i = gid.y * params.width + gid.x;
    let packed = in_rgba8.values[i];
    let r = packed & 255u;
    let g = (packed >> 8u) & 255u;
    let b = (packed >> 16u) & 255u;
    let a = f32((packed >> 24u) & 255u) * (1.0 / 255.0);
    out_linear.values[i] = vec4<f32>(
        srgb_to_linear_lut.values[r] * a,
        srgb_to_linear_lut.values[g] * a,
        srgb_to_linear_lut.values[b] * a,
        a,
    );
}
