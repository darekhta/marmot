class MarmotLm < Formula
  desc "LLM inference server and CLI powered by marmot"
  homepage "https://github.com/darekhta/marmot"
  license "MIT"
  version "0.1.0"

  on_macos do
    on_arm do
      url "https://github.com/darekhta/marmot/releases/download/v0.1.0/marmot-lm-0.1.0-darwin-arm64.tar.gz"
      sha256 "c6eebdba24e397d8d4573ea171ef3df1b7b94b3793aea3ce5e9dd45228c7ce99"
    end
    on_intel do
      url "https://github.com/darekhta/marmot/releases/download/v0.1.0/marmot-lm-0.1.0-darwin-x86_64.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/darekhta/marmot/releases/download/v0.1.0/marmot-lm-0.1.0-linux-x86_64.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  def install
    bin.install "bin/marmot-lm"
    lib.install Dir["lib/*"]

    # Ensure the binary finds the dylib via @rpath
    if OS.mac?
      MachO::Tools.change_install_name(
        "#{bin}/marmot-lm",
        "@rpath/libmarmot.dylib",
        "#{lib}/libmarmot.dylib"
      )
    end
  end

  def caveats
    <<~EOS
      marmot-lm downloads GGUF models from HuggingFace Hub on first use.
      Models are cached in ~/.local/share/marmot/models/.

      Quick start:
        marmot-lm pull bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF
        marmot-lm run TinyLlama-1.1B-Chat-v1.0 -p "Hello!"

      Start as a server:
        marmot-lm serve
    EOS
  end

  test do
    assert_match "marmot-lm", shell_output("#{bin}/marmot-lm --help")
  end
end
