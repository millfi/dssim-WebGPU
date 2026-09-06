fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VIDEO");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");

    if std::env::var_os("CARGO_FEATURE_VIDEO").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
    {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../third_party/ffmpeg-reference-shared")
            .canonicalize()
            .expect("Build Reference FFmpeg with tools/build_reference.ps1 first");
        let selected = std::env::var_os("FFMPEG_DIR")
            .map(std::path::PathBuf::from)
            .and_then(|path| path.canonicalize().ok());
        assert_eq!(
            selected.as_ref(),
            Some(&root),
            "Reference must link its own unpatched DLLs; use tools/build_reference.ps1"
        );
        let marker = root.join("dssim-ffmpeg-variant.txt");
        println!("cargo:rerun-if-changed={}", marker.display());
        assert_eq!(
            std::fs::read_to_string(marker)
                .expect("Missing FFmpeg build marker")
                .trim(),
            "reference-shared-v1",
            "Wrong FFmpeg variant for reference"
        );
        // OUT_DIR is <target>/<profile>/build/<package>/out. Keep DLLs app-local
        // for direct Cargo builds as well as the PowerShell build helper.
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        let destination = out.ancestors().nth(3).expect("Unexpected Cargo OUT_DIR");
        for entry in std::fs::read_dir(root.join("bin")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("dll") {
                println!("cargo:rerun-if-changed={}", path.display());
                std::fs::copy(&path, destination.join(path.file_name().unwrap())).unwrap();
            }
        }
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
