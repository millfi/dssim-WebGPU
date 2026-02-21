struct Vec4Buf {
    values: array<vec4<f32>>,
};

struct Params {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
};

@group(0) @binding(0) var<storage, read> in_pixels: Vec4Buf;
@group(0) @binding(1) var<storage, read_write> out_pixels: Vec4Buf;
@group(0) @binding(2) var<uniform> params: Params;

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

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let sx = u32(clamp(sx0 + dx, 0, max_x));
            let sy = u32(clamp(sy0 + dy, 0, max_y));
            let si = sy * params.in_width + sx;
            sum = sum + in_pixels.values[si];
        }
    }

    out_pixels.values[i] = sum * 0.25;
}
