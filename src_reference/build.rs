fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VIDEO");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");

    if std::env::var_os("CARGO_FEATURE_VIDEO").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
    {
        // ffmpeg-sys-next discovers prebuilt FFmpeg through FFMPEG_DIR, but
        // does not propagate the Windows system libraries required by our
        // D3D11VA-enabled FFmpeg distribution.
        for library in [
            "advapi32", "bcrypt", "d3d11", "dxgi", "ole32", "secur32", "shell32", "user32",
            "ws2_32",
        ] {
            println!("cargo:rustc-link-lib={library}");
        }
    }
}
