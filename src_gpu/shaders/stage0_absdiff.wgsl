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
@group(0) @binding(2) var<storage, read_write> out_dssim_q: U32Buf;
@group(0) @binding(3) var<storage, read_write> out_mu1: F32Buf;
@group(0) @binding(4) var<storage, read_write> out_mu2: F32Buf;
@group(0) @binding(5) var<storage, read_write> out_var1: F32Buf;
@group(0) @binding(6) var<storage, read_write> out_var2: F32Buf;
@group(0) @binding(7) var<storage, read_write> out_cov12: F32Buf;
@group(0) @binding(8) var<uniform> params: Params;

fn gaussian_weight_5x5(dx: i32, dy: i32) -> f32 {
    let ax = abs(dx);
    let ay = abs(dy);
    if (ax == 0 && ay == 0) {
        return 0.113540;
    }
    if ((ax == 1 && ay == 0) || (ax == 0 && ay == 1)) {
        return 0.079586;
    }
    if ((ax == 2 && ay == 0) || (ax == 0 && ay == 2)) {
        return 0.032123;
    }
    if (ax == 1 && ay == 1) {
        return 0.055786;
    }
    if ((ax == 2 && ay == 1) || (ax == 1 && ay == 2)) {
        return 0.022516;
    }
    return 0.009088;
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
            let w = gaussian_weight_5x5(dx, dy);

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
    let var1 = max(sumsq1 - mu1 * mu1, vec3<f32>(0.0, 0.0, 0.0));
    let var2 = max(sumsq2 - mu2 * mu2, vec3<f32>(0.0, 0.0, 0.0));
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
    let dssim = clamp(0.5 * (1.0 - ssim), 0.0, 1.0);
    let dssim_q = u32(round(dssim * f32(params.qscale)));

    out_dssim_q.values[i] = dssim_q;
    out_mu1.values[i] = mu1.x;
    out_mu2.values[i] = mu2.x;
    out_var1.values[i] = var1.x;
    out_var2.values[i] = var2.x;
    out_cov12.values[i] = cov12.x;
}
