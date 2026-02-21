struct U32Buf {
    values: array<u32>,
};

struct Params {
    len: u32,
    qscale: u32,
};

@group(0) @binding(0) var<storage, read> in1: U32Buf;
@group(0) @binding(1) var<storage, read> in2: U32Buf;
@group(0) @binding(2) var<storage, read_write> outv: U32Buf;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }

    let p1 = in1.values[i];
    let p2 = in2.values[i];

    let r1 = f32((p1 >> 0u) & 0xFFu) / 255.0;
    let g1 = f32((p1 >> 8u) & 0xFFu) / 255.0;
    let b1 = f32((p1 >> 16u) & 0xFFu) / 255.0;
    let r2 = f32((p2 >> 0u) & 0xFFu) / 255.0;
    let g2 = f32((p2 >> 8u) & 0xFFu) / 255.0;
    let b2 = f32((p2 >> 16u) & 0xFFu) / 255.0;

    let y1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
    let y2 = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2;

    let c1 = 0.01 * 0.01;
    let ssim = (2.0 * y1 * y2 + c1) / (y1 * y1 + y2 * y2 + c1);
    let dssim = clamp(0.5 * (1.0 - ssim), 0.0, 1.0);
    outv.values[i] = u32(round(dssim * f32(params.qscale)));
}
