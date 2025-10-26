use std::env;
use std::path::PathBuf;

fn main() {
    // Try pkg-config first (for installed libmarmot via brew or meson install)
    if let Ok(lib) = pkg_config::probe_library("marmot") {
        // pkg-config found — trust its flags completely.
        // Re-run if the .pc file changes (covers library rebuilds).
        for path in &lib.link_paths {
            let pc = path.join("pkgconfig/marmot.pc");
            if pc.exists() {
                println!("cargo:rerun-if-changed={}", pc.display());
            }
        }
        return;
    }

    // Fall back to local build directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let marmot_root = PathBuf::from(&manifest_dir)
        .join("../..")
        .canonicalize()
        .unwrap();

    // Check build directories in order of preference (including sanitizer variants)
    let build_dirs = [
        "build-release",
        "build-debug",
        "build-debugoptimized",
        "build-release-asan",
        "build-debug-asan",
        "build-release-ubsan",
        "build-debug-ubsan",
        "build-debug-tsan",
    ];

    let mut found_dir = None;
    for dir in &build_dirs {
        let lib_dir = marmot_root.join(dir);
        let has_lib = lib_dir.join("libmarmot.dylib").exists()
            || lib_dir.join("libmarmot.so").exists();
        if has_lib {
            found_dir = Some(lib_dir);
            break;
        }
    }

    let lib_dir = match found_dir {
        Some(dir) => dir,
        None => panic!(
            "Could not find libmarmot. Either:\n\
             1. Install it: cd {} && meson install -C build-release\n\
             2. Build it:   cd {} && make build\n\
             \n\
             pkg-config was also tried but could not find 'marmot'.",
            marmot_root.display(),
            marmot_root.display()
        ),
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=marmot");

    // Set rpath so the binary can find libmarmot at runtime
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // macOS frameworks — only link what the local build actually uses.
    // Read meson build options to determine which frameworks are needed.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=c++");

        // Accelerate is optional — check if the build was configured with it
        let intro_path = lib_dir.join("meson-info/intro-buildoptions.json");
        let link_accelerate = if intro_path.exists() {
            std::fs::read_to_string(&intro_path)
                .map(|s| s.contains("\"enable_apple_accelerate\"") && s.contains("\"value\": true"))
                .unwrap_or(true) // if we can't parse, link it to be safe
        } else {
            true // no meson info available, link to be safe
        };

        if link_accelerate {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=m");
    }

    // Include path for headers
    let include_dir = marmot_root.join("include");
    println!("cargo:include={}", include_dir.display());

    // Rerun if library or public headers change
    println!("cargo:rerun-if-changed={}", lib_dir.join("libmarmot.dylib").display());
    println!("cargo:rerun-if-changed={}", lib_dir.join("libmarmot.so").display());
    println!("cargo:rerun-if-changed={}", marmot_root.join("include/marmot").display());
}
