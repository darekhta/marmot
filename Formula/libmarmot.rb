class Libmarmot < Formula
  desc "High-performance tensor computation and ML inference library (development files)"
  homepage "https://github.com/darekhta/marmot"
  license "MIT"
  version "0.1.0"

  on_macos do
    on_arm do
      url "https://github.com/darekhta/marmot/releases/download/v0.1.0/libmarmot-dev-0.1.0-darwin-arm64.tar.gz"
      sha256 "b0e8bbbad5ffa3311ce545871d4c882e23cf7a7ec36e0fe4aeca51edded67caa"
    end
    on_intel do
      url "https://github.com/darekhta/marmot/releases/download/v0.1.0/libmarmot-dev-0.1.0-darwin-x86_64.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/darekhta/marmot/releases/download/v0.1.0/libmarmot-dev-0.1.0-linux-x86_64.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  def install
    lib.install Dir["lib/*.dylib"], Dir["lib/*.so*"], Dir["lib/*.a"]
    include.install "include/marmot"
    (lib/"pkgconfig").install Dir["lib/pkgconfig/*.pc"]
  end

  test do
    (testpath/"test.c").write <<~C
      #include <marmot/types.h>
      #include <marmot/error.h>

      int main(void) {
        marmot_dtype_t dt = MARMOT_DTYPE_FLOAT32;
        (void)dt;
        return 0;
      }
    C
    system ENV.cc, "-std=c2x", "test.c",
           "-I#{include}", "-L#{lib}", "-lmarmot",
           "-o", "test"
    system "./test"
  end
end
