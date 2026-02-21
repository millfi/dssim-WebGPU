struct Vec4Buf {
    values: array<vec4<f32>>,
};

struct Params {
    len: u32,
    width: u32,
    height: u32,
    qscale: u32,
};

@group(0) @binding(0) var<storage, read> in_pixels: Vec4Buf;
@group(0) @binding(1) var<storage, read_write> out_lab: Vec4Buf;
@group(0) @binding(2) var<uniform> params: Params;

fn cbrt_poly(x: f32) -> f32 {
    var y = (-0.5 * x + 1.51) * x + 0.2;
    var y3 = y * y * y;
    y = y * (y3 + 2.0 * x) / (2.0 * y3 + x);
    y3 = y * y * y;
    y = y * (y3 + 2.0 * x) / (2.0 * y3 + x);
    return y;
}

fn lab_from_rgbaplu(px: vec4<f32>, x: i32, y: i32) -> vec3<f32> {
    var r = px.x;
    var g = px.y;
    var b = px.z;
    let a = px.w;

    // Match dssim-core ToRGB for RGBAPLU pixels.
    let n = u32((x + 11) ^ (y + 11));
    if (a < 255.0) {
        let one_minus_a = 1.0 - a;
        if ((n & 16u) != 0u) {
            r = r + one_minus_a;
        }
        if ((n & 8u) != 0u) {
            g = g + one_minus_a;
        }
        if ((n & 32u) != 0u) {
            b = b + one_minus_a;
        }
    }

    let d65x = 0.9505;
    let d65y = 1.0;
    let d65z = 1.089;
    let fx = r * (0.4124 / d65x) + g * (0.3576 / d65x) + b * (0.1805 / d65x);
    let fy = r * (0.2126 / d65y) + g * (0.7152 / d65y) + b * (0.0722 / d65y);
    let fz = r * (0.0193 / d65z) + g * (0.1192 / d65z) + b * (0.9505 / d65z);

    let epsilon = 216.0 / 24389.0;
    let k = 24389.0 / (27.0 * 116.0);
    let X = select(k * fx, cbrt_poly(fx) - 16.0 / 116.0, fx > epsilon);
    let Y = select(k * fy, cbrt_poly(fy) - 16.0 / 116.0, fy > epsilon);
    let Z = select(k * fz, cbrt_poly(fz) - 16.0 / 116.0, fz > epsilon);

    let l = Y * 1.05;
    let a2 = (500.0 / 220.0) * (X - Y) + (86.2 / 220.0);
    let b2 = (200.0 / 220.0) * (Y - Z) + (107.9 / 220.0);
    return vec3<f32>(l, a2, b2);
}

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

    let center = lab_from_rgbaplu(in_pixels.values[i], x, y);

    var pre_a = 0.0;
    var pre_b = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let nx = clamp(x + dx, 0, max_x);
            let ny = clamp(y + dy, 0, max_y);
            let ni = u32(ny) * params.width + u32(nx);
            let w = gaussian_weight_5x5(dx, dy);
            let lab = lab_from_rgbaplu(in_pixels.values[ni], nx, ny);
            pre_a = pre_a + w * lab.y;
            pre_b = pre_b + w * lab.z;
        }
    }

    out_lab.values[i] = vec4<f32>(center.x, pre_a, pre_b, 0.0);
}
