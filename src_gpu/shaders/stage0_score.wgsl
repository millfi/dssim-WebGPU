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
    qscale: u32,
};

@group(0) @binding(0) var<storage, read> in1: Vec4Buf;
@group(0) @binding(1) var<storage, read> in2: Vec4Buf;
@group(0) @binding(2) var<storage, read_write> out_ssim: F32Buf;
@group(0) @binding(3) var<uniform> params: Params;

const gaussian_weights = array<f32, 25>(
    0.009088, 0.022516, 0.032123, 0.022516, 0.009088,
    0.022516, 0.055786, 0.079586, 0.055786, 0.022516,
    0.032123, 0.079586, 0.113540, 0.079586, 0.032123,
    0.022516, 0.055786, 0.079586, 0.055786, 0.022516,
    0.009088, 0.022516, 0.032123, 0.022516, 0.009088,
);

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let i = gid.y * params.width + gid.x;

    let x = i32(gid.x);
    let y = i32(gid.y);
    let max_x = i32(params.width) - 1;
    let max_y = i32(params.height) - 1;

    var sum1 = vec3<f32>(0.0);
    var sum2 = vec3<f32>(0.0);
    var sumsq1 = 0.0;
    var sumsq2 = 0.0;
    var sum12 = 0.0;

    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let nx = clamp(x + dx, 0, max_x);
            let ny = clamp(y + dy, 0, max_y);
            let ni = u32(ny) * params.width + u32(nx);
            let w = gaussian_weights[u32((dy + 2) * 5 + dx + 2)];
            let lab1 = in1.values[ni].xyz;
            let lab2 = in2.values[ni].xyz;

            sum1 = sum1 + w * lab1;
            sum2 = sum2 + w * lab2;
            sumsq1 = sumsq1 + w * dot(lab1, lab1);
            sumsq2 = sumsq2 + w * dot(lab2, lab2);
            sum12 = sum12 + w * dot(lab1, lab2);
        }
    }

    let mu1_sq = dot(sum1, sum1) / 3.0;
    let mu2_sq = dot(sum2, sum2) / 3.0;
    let mu1_mu2 = dot(sum1, sum2) / 3.0;
    let sigma1_sq = sumsq1 / 3.0 - mu1_sq;
    let sigma2_sq = sumsq2 / 3.0 - mu2_sq;
    let sigma12 = sum12 / 3.0 - mu1_mu2;

    let c1 = 0.01 * 0.01;
    let c2 = 0.03 * 0.03;
    let numer = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2);
    let denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2);
    out_ssim.values[i] = numer / denom;
}
