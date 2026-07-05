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
    qscale: u32,
};

@group(0) @binding(0) var<storage, read> in1: Vec4Buf;
@group(0) @binding(1) var<storage, read> in2: Vec4Buf;
@group(0) @binding(2) var<storage, read_write> out_ssim: F32Buf;
@group(0) @binding(3) var<storage, read_write> out_mu1: F32Buf;
@group(0) @binding(4) var<storage, read_write> out_mu2: F32Buf;
@group(0) @binding(5) var<storage, read_write> out_var1: F32Buf;
@group(0) @binding(6) var<storage, read_write> out_var2: F32Buf;
@group(0) @binding(7) var<storage, read_write> out_cov12: F32Buf;
@group(0) @binding(8) var<uniform> params: Params;

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

    var sum1 = vec3<f32>(0.0, 0.0, 0.0);
    var sum2 = vec3<f32>(0.0, 0.0, 0.0);
    var sumsq1 = vec3<f32>(0.0, 0.0, 0.0);
    var sumsq2 = vec3<f32>(0.0, 0.0, 0.0);
    var sum12 = vec3<f32>(0.0, 0.0, 0.0);

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
            sumsq1 = sumsq1 + w * lab1 * lab1;
            sumsq2 = sumsq2 + w * lab2 * lab2;
            sum12 = sum12 + w * lab1 * lab2;
        }
    }

    let mu1 = sum1;
    let mu2 = sum2;
    let var1 = sumsq1 - mu1 * mu1;
    let var2 = sumsq2 - mu2 * mu2;
    let cov12 = sum12 - mu1 * mu2;

    let mu1_sq = (mu1.x * mu1.x + mu1.y * mu1.y + mu1.z * mu1.z) / 3.0;
    let mu2_sq = (mu2.x * mu2.x + mu2.y * mu2.y + mu2.z * mu2.z) / 3.0;
    let mu1_mu2 = (mu1.x * mu2.x + mu1.y * mu2.y + mu1.z * mu2.z) / 3.0;
    let sigma1_sq = (var1.x + var1.y + var1.z) / 3.0;
    let sigma2_sq = (var2.x + var2.y + var2.z) / 3.0;
    let sigma12 = (cov12.x + cov12.y + cov12.z) / 3.0;

    let c1 = 0.01 * 0.01;
    let c2 = 0.03 * 0.03;
    let numer = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2);
    let denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2);
    let ssim = numer / denom;

    out_ssim.values[i] = ssim;
    out_mu1.values[i] = mu1.x;
    out_mu2.values[i] = mu2.x;
    out_var1.values[i] = var1.x;
    out_var2.values[i] = var2.x;
    out_cov12.values[i] = cov12.x;
}
