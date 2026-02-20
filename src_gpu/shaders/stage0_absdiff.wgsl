struct U32Buf {
    values: array<u32>,
};

struct Params {
    len: u32,
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

    let a = in1.values[i];
    let b = in2.values[i];
    outv.values[i] = select(b - a, a - b, a >= b);
}
