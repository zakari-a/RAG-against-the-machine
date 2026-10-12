with open("data/raw/vllm-0.10.1/docs/features/sleep_mode.md", "r") as file:
    text = file.read()
    texts = text.split("\n")
    for t in texts:
        if t.startswith("#"):
            print(t)