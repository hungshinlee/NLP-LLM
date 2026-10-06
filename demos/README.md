<!-- 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。 -->
# Classroom demos — code and rehearsal records

These are the programs the instructor runs live in class. **You are not required to run anything**: the course is lectures only, and every number you see on the slides comes from the rehearsal records checked in here (`runs/rehearsal/` in each folder). The code is published so that you can read it, re-run it, and change it if you want to.

Each week's page on the course site has a *Demos* section that says what each demo shows and links to its folder here. Each folder has its own `README.md` with requirements, setup, how to run it, and what you should see.

## Layout

```
demos/
  common.py                 small helpers shared by all demos (terminal colours, version lock, memory probes)
  setup.sh                  builds the shared .venv on Apple-silicon macOS and pins versions (see below)
  versions.lock             pinned versions: mlx, mlx-lm git commit, transformers, torch, and every model revision
  requirements.lock.txt     full `pip freeze` of the instructor's environment, for reference
  wNN_dK_<name>/            one demo per folder: week NN, demo K (the "Demo K" in the slide footers)
    demo.py / *.py          the program
    present.sh              entry point (always offline: HF_HUB_OFFLINE=1)
    demo_config.toml        model, parameters, texts — change these, not the code
    runs/rehearsal/         the instructor's rehearsal record: the numbers on the slides
    README.md               what it shows, requirements, how to run, what to expect
```

## The machine these were tested on

**Everything here was run on one machine only: a MacBook Pro with an Apple M5 Max (64 GB unified memory, no CUDA), macOS 26.6, Python 3.12.** The per-demo READMEs say which demos are portable and which are not:

| Needs | Demos | Runs on |
|---|---|---|
| `torch` only, CPU | W3 demo 1; W4 demo 1 (tests, `count`) | any laptop |
| `torch` only, training on `mps` | W4 demo 1 (`rehearse`), W4 demo 2 (`rehearse`) | Apple silicon as tested; falls back to CPU (slow); `cuda` accepted by the code but never run. `replay` needs no device at all |
| `torch` + `transformers` + a small Hugging Face model | W3 demo 3 (CPU), W4 demo 3 (`mps`, or `--device cpu`) | any laptop with ~4 GB free RAM and a network connection for the first download |
| `mlx` + `mlx-lm` | W3 demo 2 | **Apple silicon only** |

Two cross-folder dependencies to know about: W4 demos 1 and 2 import W3 demo 1's `attention_final.py` and each other by relative path, and W4 demo 3 reads two short texts from `w01_d2_bpb/texts/` and `w02_d1_fertility/texts/` (published here even though those two demos are not yet) — keep the folder layout as it is. W4 demos 1 and 2 train on `tinyshakespeare.txt`, which is not in the repository; W4 demo 1's README says where to download it and where to put it.

Nothing has been tested on Linux, Windows, or a CUDA GPU. Where a README says "should also work on …", read that as a statement of what the code *does not* depend on, not as something that was verified.

## Setting up

### Apple-silicon Mac (same path as the instructor)

```bash
cd demos
./setup.sh --no-fetch      # creates demos/.venv with the pinned mlx, mlx-lm, transformers, torch (needs `uv`: brew install uv)
```

Then follow the per-demo README; demos that need a model have a `./present.sh fetch` step.

### Anything else (Linux, Windows/WSL, Intel Mac)

`setup.sh` installs `mlx`, which only exists for Apple silicon, so it will fail. Build the environment by hand with just the two packages the portable demos need:

```bash
cd demos
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.14.0 transformers==5.17.0
```

(`pip` works too: `python3.12 -m venv .venv && .venv/bin/pip install torch==2.14.0 transformers==5.17.0`.) The `present.sh` scripts look for `../.venv/bin/python`, so the folder has to be called `.venv` and live directly under `demos/`.

The versions are the ones in `versions.lock`; the code was not tested with any other.

## Models

Models are **not** in this repository. They are downloaded into the Hugging Face cache (`~/.cache/huggingface/hub` by default, or wherever `HF_HOME` points) by each demo's `fetch` command, which also pins the exact model revision in `versions.lock`. `present.sh` runs everything with `HF_HUB_OFFLINE=1`, so after the first `fetch` nothing touches the network.

## Two things to know about the output

- Prompts and labels printed to the terminal are in Chinese (the course is taught in Chinese; slides are in English). The keys are always the same: **Enter** advances, **q** quits, and demo 3 uses **n** for the next sentence.
- Every `replay` command shows the rehearsal record, not a live run, and the screen says so. Numbers from a `replay` are the ones on the slides; numbers from a live run on your machine will differ in timing and, for memory, in the fixed overhead — the formula-vs-measurement comparison is the point, not the exact figure.
