# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""W7 demo 4 的三個測試（投影片 p46 前導頁 What runs 那列的 (i)–(iii)）。dedup_final.py 與 dedup_live.py 共用。

    python3 tests.py dedup_live           # 課堂：對 live 版跑（等同 ./present.sh live）；還沒寫的函數會標 skip
    python3 tests.py dedup_final          # 等同 ./present.sh test
    python3 tests.py --replay             # 顯示彩排紀錄 runs/rehearsal/tests.json

寫到哪測到哪：
  (i)   需要 shingles；有 minhash 就多驗一件事      兩份一樣的文件：精確 Jaccard = 1，MinHash 估計 = 1（128 個簽章全部相等）
  (ii)  需要 minhash                                 已知 Jaccard 的人造 shingle 集合（J = 0.75 與 0.30，各 1,000 個元素）：
                                                     估計落在 J ± 2/sqrt(k_h)（= ±0.177）內；另印 ± 一個標準差 sqrt(J(1−J)/k_h)
  (iii) 需要 minhash 與 lsh_candidates               60 對人造文件、J 從 0.10 到 0.99：所有 J > 0.9 的對都要是 (b, r) = (16, 8) 的候選；
                                                     另印每個 J 檔實際成為候選的比例，對 S 曲線 1 − (1 − J^r)^b
純標準函式庫；人造集合的元素是字串（"s000123"），不經過 shingles。
"""
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "rehearsal"
BOLD, DIM, RESET, GREEN, RED, AMBER, GREY = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[31m", "\033[33m", "\033[90m"

K_H = 128
B_LSH, R_LSH = 16, 8
N_SHINGLE = 5
SIZE = 1000                                             # (ii)(iii) 每個集合的元素數
J_PAIRS_II = [0.75, 0.30]
J_LEVELS_III = [0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]   # 每檔幾對見 PAIRS_PER_LEVEL
PAIRS_PER_LEVEL = {0.10: 6, 0.30: 6, 0.50: 6, 0.60: 8, 0.70: 8, 0.80: 8, 0.90: 6, 0.95: 6, 0.99: 6}   # 共 60 對
TEXT_A = ("Deduplication removes documents that are copies or near copies of one another. "
          "A duplicate is a token predicted many times, so the model memorises instead of generalising. "
          "MinHash estimates the Jaccard similarity of two sets from a short signature; locality sensitive hashing finds the candidate pairs without comparing every pair.")


def _has(impl, name):
    return callable(getattr(impl, name, None))


def _known_pair(J, size, tag):
    """兩個集合，各 size 個元素，Jaccard 剛好 J：共用 s 個、各自獨有 u 個，s/(s + 2u) = J → s = round(size·2J/(1+J))。"""
    s = round(size * 2 * J / (1 + J))
    u = size - s
    shared = {f"{tag}-s{i:05d}" for i in range(s)}
    A = shared | {f"{tag}-a{i:05d}" for i in range(u)}
    B = shared | {f"{tag}-b{i:05d}" for i in range(u)}
    return A, B, s / (s + 2 * u)


def _jaccard(A, B):
    u = len(A | B)
    return len(A & B) / u if u else 0.0


def _estimate(x, y):
    return sum(p == q for p, q in zip(x, y)) / len(x)


# ── (i) 一樣的文件 ──────────────────────────────────────────────────
def test_i(impl):
    A, B = impl.shingles(TEXT_A, N_SHINGLE), impl.shingles(TEXT_A, N_SHINGLE)
    n_words = len(TEXT_A.split())
    nums = {"words": n_words, "shingles": len(A), "exact_J": _jaccard(A, B)}
    if not isinstance(A, (set, frozenset)):
        return "FAIL", f"shingles 回傳的是 {type(A).__name__}，要是 set", nums
    if nums["exact_J"] != 1.0:
        return "FAIL", f"同一份文件的精確 Jaccard = {nums['exact_J']}", nums
    msg = f"{n_words} words → {len(A)} shingles (n = {N_SHINGLE}); exact J = 1.0"
    if _has(impl, "minhash"):
        sa, sb = impl.minhash(A, K_H), impl.minhash(B, K_H)
        if sa is None or len(sa) != K_H:
            return "FAIL", f"minhash 回傳長度 {None if sa is None else len(sa)}，要是 {K_H}", nums
        nums["estimate"] = _estimate(sa, sb)
        nums["k_h"] = K_H
        msg += f"; MinHash estimate = {nums['estimate']:.3f} ({sum(p == q for p, q in zip(sa, sb))}/{K_H} signatures equal)"
        if nums["estimate"] != 1.0:
            return "FAIL", msg, nums
    else:
        msg += f"  {GREY}(minhash 還沒寫：只驗了精確值){RESET}"
    return "ok", msg, nums


# ── (ii) 已知 Jaccard 的人造對 ──────────────────────────────────────
def test_ii(impl):
    tol = 2 / math.sqrt(K_H)
    rows, ok = [], True
    for J in J_PAIRS_II:
        A, B, J_exact = _known_pair(J, SIZE, f"ii{int(J * 100)}")
        est = _estimate(impl.minhash(A, K_H), impl.minhash(B, K_H))
        sd = math.sqrt(J_exact * (1 - J_exact) / K_H)
        inside = abs(est - J_exact) <= tol
        ok &= inside
        rows.append({"J": J_exact, "estimate": est, "error": est - J_exact, "sd": sd, "tol": tol, "inside": inside})
    msg = "; ".join(f"J = {r['J']:.2f}: estimate {r['estimate']:.3f} ({r['error']:+.3f}; sd {r['sd']:.3f}, tol ±{r['tol']:.3f}) {'ok' if r['inside'] else 'OUT'}" for r in rows)
    return ("ok" if ok else "FAIL"), msg, {"k_h": K_H, "size": SIZE, "pairs": rows}


# ── (iii) 候選對包含所有 J > 0.9 的對 ───────────────────────────────
def test_iii(impl):
    sets, pairs = [], []                                   # sets 的索引是「文件編號」
    for J in J_LEVELS_III:
        for t in range(PAIRS_PER_LEVEL[J]):
            A, B, J_exact = _known_pair(J, SIZE, f"iii{int(J * 100)}-{t}")
            pairs.append((len(sets), len(sets) + 1, J_exact))
            sets += [A, B]
    sigs = [impl.minhash(S, K_H) for S in sets]
    cands = impl.lsh_candidates(sigs, B_LSH, R_LSH)
    cands = {tuple(sorted(p)) for p in cands}
    by_level = {}
    for i, j, J in pairs:
        lv = by_level.setdefault(round(J, 2), {"pairs": 0, "candidates": 0})
        lv["pairs"] += 1
        lv["candidates"] += (i, j) in cands
    for J, lv in by_level.items():
        lv["fraction"] = lv["candidates"] / lv["pairs"]
        lv["s_curve"] = 1 - (1 - J ** R_LSH) ** B_LSH
    high = [(i, j, J) for i, j, J in pairs if J > 0.9]
    missed = [(i, j, J) for i, j, J in high if (i, j) not in cands]
    extra = len(cands) - sum(((i, j) in cands) for i, j, _ in pairs)      # 不是人造對的候選（不同對之間的誤撞）
    table = "  ".join(f"J={J:.2f}:{lv['candidates']}/{lv['pairs']} (S {lv['s_curve']:.2f})" for J, lv in sorted(by_level.items()))
    msg = f"{len(pairs)} pairs, {len(cands)} candidate pairs at (b, r) = ({B_LSH}, {R_LSH}); all {len(high)} pairs with J > 0.9 are candidates" if not missed else f"{len(missed)} of {len(high)} pairs with J > 0.9 are NOT candidates"
    msg += f"; cross-pair false positives: {extra}\n      per level (candidates/pairs, S-curve): {table}"
    nums = {"k_h": K_H, "b": B_LSH, "r": R_LSH, "pairs": len(pairs), "candidates": len(cands), "missed_high": len(missed), "cross_pair_fp": extra,
            "levels": {f"{J:.2f}": lv for J, lv in sorted(by_level.items())}}
    return ("ok" if not missed else "FAIL"), msg, nums


TESTS = [("i", test_i, ("shingles",)), ("ii", test_ii, ("minhash",)), ("iii", test_iii, ("minhash", "lsh_candidates"))]


def run_all(impl):
    results = {}
    for name, fn, needs in TESTS:
        missing = [n for n in needs if not _has(impl, n)]
        if missing:
            print(f"  ({name}) {GREY}skip{RESET}  還沒寫 {', '.join(missing)}")
            results[name] = {"status": "skip", "missing": missing}
            continue
        t0 = time.perf_counter()
        try:
            status, msg, nums = fn(impl)
        except Exception as e:                           # 現場打錯字是預期內的事：印一行，不炸
            status, msg, nums = "FAIL", f"{type(e).__name__}: {e}", {}
        dt = time.perf_counter() - t0
        color = GREEN if status == "ok" else RED
        print(f"  ({name}) {color}{status}{RESET}  {msg}  {GREY}{dt:.2f} s{RESET}")
        results[name] = {"status": status, "seconds": dt, **nums}
    return results


def replay():
    p = RUNS / "tests.json"
    if not p.exists():
        raise SystemExit(f"{RED}沒有 {p}：先跑 ./present.sh rehearse{RESET}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    print(f"{AMBER}{BOLD}── 彩排紀錄（非現場）{RESET} {GREY}{rec['date']} · {rec['python']} · {rec['machine']}{RESET}")
    for name, r in rec["tests"].items():
        color = GREEN if r.get("status") == "ok" else (GREY if r.get("status") == "skip" else RED)
        print(f"  ({name}) {color}{r.get('status')}{RESET}  " + json.dumps({k: v for k, v in r.items() if k not in ("status", "seconds", "levels", "pairs")}, ensure_ascii=False))
        if "levels" in r:
            print("        " + "  ".join(f"J={J}:{lv['candidates']}/{lv['pairs']} (S {lv['s_curve']:.2f})" for J, lv in r["levels"].items()))


def main():
    if "--replay" in sys.argv:
        return replay()
    mod = next((a for a in sys.argv[1:] if not a.startswith("-")), "dedup_final")
    sys.path.insert(0, str(HERE))
    impl = __import__(mod)
    print(f"{BOLD}W7 demo 4 · tests against {mod}.py{RESET}  {GREY}k_h = {K_H}, (b, r) = ({B_LSH}, {R_LSH}), n = {N_SHINGLE}{RESET}")
    results = run_all(impl)
    if "--save" in sys.argv:
        RUNS.mkdir(parents=True, exist_ok=True)
        rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}", "impl": mod, "tests": results}
        (RUNS / "tests.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{GREY}→ {RUNS / 'tests.json'}{RESET}")
    return 0 if all(r.get("status") in ("ok", "skip") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
