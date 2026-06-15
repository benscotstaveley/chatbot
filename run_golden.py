from config import Config
import subprocess


def main():

    config = Config.load()
    print(config)

    cmd=[
        "../../../../llama.cpp-latest/build/bin/llama-cli",
        "-m", config.model,
        "--n-gpu-layers", str(config.ngl),
        "--ctx-size", str(config.ctx),
        "--temp", str(config.temp),
        "--top-p", str(config.top_p),
        "--top-k", str(config.top_k),
        "--min-p", str(config.min_p),
        "--typical-p", str(config.typical_p),
        "--repeat-penalty", str(config.repeat_penalty),
        "-no-cnv",
        "-f", "rendered_prompt.txt",
        "--seed", str(config.seed)
        ]

    if config.flash_attn:
        cmd.append("--flash-attn")
        cmd.append("on")
    if False:
        cmd.append("--verbose-prompt")
        cmd.append("-v")

    print(f"\nAbout to execute: -{" ".join(cmd)}-")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        with open("golden.log", "w") as f:
            f.write(result.stdout)

    except subprocess.CalledProcessError as e:
        print("Command failed!")
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)    

# end of main


if __name__ == "__main__":
    main()
