struct U32Buf {
    values: array<u32>,
};

struct Params {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
};

@group(0) @binding(0) var<storage, read> in_pixels: U32Buf;
@group(0) @binding(1) var<storage, read_write> out_pixels: U32Buf;
@group(0) @binding(2) var<uniform> params: Params;

fn unpack_r(p: u32) -> u32 {
    return (p >> 0u) & 0xFFu;
}
fn unpack_g(p: u32) -> u32 {
    return (p >> 8u) & 0xFFu;
}
fn unpack_b(p: u32) -> u32 {
    return (p >> 16u) & 0xFFu;
}
fn unpack_a(p: u32) -> u32 {
    return (p >> 24u) & 0xFFu;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_len = params.out_width * params.out_height;
    if (i >= out_len) {
        return;
    }

    let ox = i % params.out_width;
    let oy = i / params.out_width;
    let sx0 = i32(ox * 2u);
    let sy0 = i32(oy * 2u);
    let max_x = i32(params.in_width) - 1;
    let max_y = i32(params.in_height) - 1;

    var sum_r: u32 = 0u;
    var sum_g: u32 = 0u;
    var sum_b: u32 = 0u;
    var sum_a: u32 = 0u;

    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let sx = u32(clamp(sx0 + dx, 0, max_x));
            let sy = u32(clamp(sy0 + dy, 0, max_y));
            let si = sy * params.in_width + sx;
            let p = in_pixels.values[si];
            sum_r = sum_r + unpack_r(p);
            sum_g = sum_g + unpack_g(p);
            sum_b = sum_b + unpack_b(p);
            sum_a = sum_a + unpack_a(p);
        }
    }

    let r = (sum_r + 2u) / 4u;
    let g = (sum_g + 2u) / 4u;
    let b = (sum_b + 2u) / 4u;
    let a = (sum_a + 2u) / 4u;
    out_pixels.values[i] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
