struct U32Buf {
    values: array<u32>,
};

struct Params {
    len: u32,
    width: u32,
    height: u32,
    qscale: u32,
};

@group(0) @binding(0) var<storage, read> in1: U32Buf;
@group(0) @binding(1) var<storage, read> in2: U32Buf;
@group(0) @binding(2) var<storage, read_write> outv: U32Buf;
@group(0) @binding(3) var<uniform> params: Params;

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn luma_from_rgba8_packed(p: u32) -> f32 {
    let r = srgb_to_linear(f32((p >> 0u) & 0xFFu) / 255.0);
    let g = srgb_to_linear(f32((p >> 8u) & 0xFFu) / 255.0);
    let b = srgb_to_linear(f32((p >> 16u) & 0xFFu) / 255.0);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }

    let x = i32(i % params.width);
    let y = i32(i / params.width);
    let max_x = i32(params.width) - 1;
    let max_y = i32(params.height) - 1;

    var sum1 = 0.0;
    var sum2 = 0.0;
    var sumsq1 = 0.0;
    var sumsq2 = 0.0;
    var sum12 = 0.0;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let nx = clamp(x + dx, 0, max_x);
            let ny = clamp(y + dy, 0, max_y);
            let ni = u32(ny) * params.width + u32(nx);

            let y1 = luma_from_rgba8_packed(in1.values[ni]);
            let y2 = luma_from_rgba8_packed(in2.values[ni]);

            sum1 = sum1 + y1;
            sum2 = sum2 + y2;
            sumsq1 = sumsq1 + y1 * y1;
            sumsq2 = sumsq2 + y2 * y2;
            sum12 = sum12 + y1 * y2;
        }
    }

    let n = 9.0;
    let mu1 = sum1 / n;
    let mu2 = sum2 / n;
    let var1 = max(sumsq1 / n - mu1 * mu1, 0.0);
    let var2 = max(sumsq2 / n - mu2 * mu2, 0.0);
    let cov12 = sum12 / n - mu1 * mu2;

    let c1 = 0.01 * 0.01;
    let c2 = 0.03 * 0.03;
    let numer = (2.0 * mu1 * mu2 + c1) * (2.0 * cov12 + c2);
    let denom = (mu1 * mu1 + mu2 * mu2 + c1) * (var1 + var2 + c2);
    let ssim = numer / denom;
    let dssim = clamp(0.5 * (1.0 - ssim), 0.0, 1.0);
    outv.values[i] = u32(round(dssim * f32(params.qscale)));
}
