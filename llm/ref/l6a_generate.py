#!/usr/bin/env python3
"""l6a_generate.py (2026-11-19, CPU only): greedy SmolLM2-135M-Instruct answers for the drafter study (L6a).

With greedy verification, speculative decoding produces exactly the target model's greedy output, so a drafter can
be judged offline: replay it over the saved greedy answers. "train" prompts warm the drafter's persistent tables,
"test" prompts measure it. Categories: qa (open questions), list, rewrite (the answer may copy the given text),
code (edit given code). fp32 target (the NPU-format target's greedy chain differs slightly; to be checked later).
Writes npu-llm/runs/l6a-greedy.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smollm import SmolLM, Quant, chat_ids, generate  # noqa: E402
from l2_quality import MD  # noqa: E402

PARA = [
    "The Amazon rainforest covers much of the Amazon basin of South America. It spans nine countries and holds about "
    "ten percent of all known species. Deforestation, mostly for cattle ranching and farming, has removed about a "
    "fifth of the forest in the last fifty years.",
    "Honey bees live in colonies with one queen, thousands of female workers and a few hundred male drones. Workers "
    "collect nectar and pollen, build wax combs and care for the young. A colony can make more than twenty kilograms "
    "of surplus honey in a good year.",
    "The first railways used horses to pull wagons along wooden rails. Steam locomotives arrived in the early "
    "nineteenth century and made travel over land much faster. By 1900 railways connected most large cities in "
    "Europe and North America.",
    "Volcanoes form where molten rock reaches the surface of the Earth. Most of them sit along the edges of tectonic "
    "plates, especially around the Pacific Ocean. An eruption can release lava, ash and gases that change the "
    "weather for years.",
]
CODE = [
    "def area(w, h):\n    return w * h\n\nprint(area(3, 4))",
    "numbers = [3, 1, 2]\nfor n in numbers:\n    print(n)",
    "def greet(name):\n    print('Hello ' + name)\n\ngreet('Sam')",
    "total = 0\nfor i in range(10):\n    total = total + i\nprint(total)",
]
TRAIN = [
    ("qa", "Why is the sky blue?"), ("qa", "How does a bicycle stay upright?"),
    ("qa", "What causes the seasons on Earth?"), ("qa", "How do plants make their food?"),
    ("qa", "What is the difference between weather and climate?"), ("qa", "Why do we need sleep?"),
    ("qa", "How does a microwave oven heat food?"), ("qa", "What is a black hole?"),
    ("list", "List five healthy breakfast ideas."), ("list", "Give me four tips for saving money."),
    ("list", "Name five animals that live in the ocean."), ("list", "List three ways to reduce stress."),
    ("list", "Give me five ideas for a rainy day."), ("list", "List four benefits of reading books."),
    ("rewrite", "Summarize this text in two sentences:\n" + PARA[0]),
    ("rewrite", "Rewrite this text in simpler words:\n" + PARA[1]),
    ("rewrite", "Summarize this text in one sentence:\n" + PARA[2]),
    ("rewrite", "Rewrite this text as a short list of facts:\n" + PARA[3]),
    ("code", "Add a docstring to this Python function:\n" + CODE[0]),
    ("code", "Change this code so it prints the numbers in sorted order:\n" + CODE[1]),
    ("code", "Rewrite this function to use an f-string:\n" + CODE[2]),
    ("code", "Explain what this code does:\n" + CODE[3]),
    ("qa", "What is photosynthesis?"), ("qa", "How do vaccines work?"),
    ("list", "List five common kitchen tools."), ("qa", "Why does ice float on water?"),
    ("rewrite", "Rewrite this text for a ten-year-old:\n" + PARA[0]),
    ("code", "Add comments to this code:\n" + CODE[1]),
]
TEST = [
    ("qa", "Why do leaves change color in autumn?"), ("qa", "How does the heart pump blood?"),
    ("qa", "What is gravity?"), ("qa", "How are rainbows formed?"), ("qa", "Why do cats purr?"),
    ("list", "List five tips for a good job interview."), ("list", "Name four planets in our solar system."),
    ("list", "Give me five ideas for a birthday party."), ("list", "List three ways to stay healthy."),
    ("rewrite", "Summarize this text in two sentences:\n" + PARA[1]),
    ("rewrite", "Rewrite this text in simpler words:\n" + PARA[2]),
    ("rewrite", "Summarize this text in one sentence:\n" + PARA[3]),
    ("rewrite", "Rewrite this text as a short list of facts:\n" + PARA[0]),
    ("rewrite", "Rewrite this text for a ten-year-old:\n" + PARA[3]),
    ("code", "Add a docstring to this Python function:\n" + CODE[2]),
    ("code", "Change this code so it prints the total at each step:\n" + CODE[3]),
    ("code", "Rewrite this function to use an f-string:\n" + CODE[0].replace("return w * h", "print('Area: ' + str(w * h))")),
    ("code", "Explain what this code does:\n" + CODE[1]),
    ("code", "Add comments to this code:\n" + CODE[2]),
    ("qa", "What is the water cycle?"),
]


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    m = SmolLM(MD)
    out = []
    t0 = time.monotonic()
    for split, items in (("train", TRAIN), ("test", TEST)):
        for cat, p in items:
            ids = chat_ids(tok, p)
            t = time.monotonic()
            gen = generate(m, ids, 128, Quant("fp32"))
            out.append({"split": split, "cat": cat, "prompt": p, "prompt_ids": ids, "out_ids": gen})
            print("%-5s %-7s %3d prompt + %3d out tokens %5.1f s  %r" % (split, cat, len(ids), len(gen),
                  time.monotonic() - t, tok.decode(gen)[:70]), flush=True)
    path = "/home/jordan/Documents/AI_Projects/npu-llm/runs/l6a-greedy.json"
    json.dump(out, open(path, "w"))
    print("wrote %s (%d answers, %.0f s)" % (path, len(out), time.monotonic() - t0))


if __name__ == "__main__":
    sys.exit(main())
