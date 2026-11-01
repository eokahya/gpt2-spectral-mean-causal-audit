"""v3 stages: prepare | verify-protected | verify-v2 | smoke | register | develop | power | report (DEV_PROTOCOL_v3.md).

Technical remeasurement. Every activation extraction uses the CANONICAL layout: one prompt replicated
`extraction.rows` (= 19) times, no padding, no mixing of prompts; the recorded branch input and unpatched reference
come from row `extraction.reference_row` (= 17 = the identity row of the group forward). The group forward is the
same physical layout (19 copies of the recipient). prepare copies the registered v2 manifests byte-identically
(tokenizer re-check only). smoke uses OLD Day-1 calibration prompts only. develop and power run only after the
write-once v3 registration."""
import copy
import json
import time

import numpy as np
import torch

from scp.common import get_device, memory_snapshot, sha256_file, sync, utcnow
from scp.data import make_batch, read_csv, year_token_ids
from scp.model import load_model, model_file_info

from . import ROOT
from .bootstrap import confirm_decision, template_bootstrap
from .common import (PROTECTED, RUNS, GRun, bound_hashes, ids_sha, load_run, protected_hashes, registration_path,
                     verify_protected, verify_v2_preservation, write_once)
from .data import load_groups, prepare as prepare_manifests, write_csv
from .estimators import centred_J, frame_moments, norm2_m, scale_s2, template_T, u_pairs, u_variance
from .geometry import hybrids, load_projector, support
from .hook import branch_mlp, forward
from .power import run_power

F64 = torch.float64
REGISTRATION = None  # test override; otherwise the config's registration path (registration/DEV_REGISTRATION_v3.json)
ROW_KINDS = [f"minus{k}" for k in range(8)] + [f"plus{k}" for k in range(8)] + ["nat_a", "nat_r", "dup_minus0"]


class ResourceStop(RuntimeError):
    def __init__(self, status, done):
        super().__init__(status)
        self.status, self.done = status, done


def check_canonical(batch, ids, rows):
    """The canonical layout: `rows` identical copies of ONE prompt, no padding, every mask 1, last index len-1."""
    n = len(ids)
    ok = (tuple(batch.ids.shape) == (rows, n) and bool((batch.mask == 1).all())
          and bool((batch.last_idx == n - 1).all())
          and bool((batch.ids == torch.as_tensor(ids, device=batch.ids.device)[None]).all()))
    if not ok:
        raise AssertionError("non-canonical layout (padding, mixing or wrong row count)")
    return True


def extraction_layout(cfg):
    e = cfg["extraction"]
    if e["layout"] != "canonical_no_padding_replicated" or e["reference_row"] != ROW_KINDS.index("nat_r") \
            or e["rows"] != len(ROW_KINDS):
        raise ValueError("extraction layout must be canonical, 19 rows, reference row 17 (= the identity row)")
    return e["rows"], e["reference_row"]


def _staged(stage, cfg, fn, extra=None):
    run = GRun(stage, cfg, extra)
    try:
        res = fn(run)
        run.finish("ok")
        return run, res
    except Exception as e:  # failed runs are kept
        run.finish("failed", e)
        raise


# ================================================================ prepare / protected

def prepare(cfg):
    def go(run):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        J = year_token_ids(tok)
        if ids_sha(J) != cfg["observable"]["J_ids_sha256"] or len(set(J)) != cfg["observable"]["m"]:
            raise ValueError("J token list differs from the registered list")
        if not PROTECTED.exists():
            write_once(PROTECTED, {"created_utc": utcnow().isoformat(), **protected_hashes()})
        P, pchk = load_projector(cfg)
        idx = prepare_manifests(tok, cfg)
        seeds = {"main": cfg["seed"], "build": cfg["seed"] + 1, "power": cfg["power"]["seed"],
                 "smoke": cfg["smoke"]["seed"], "dev_bootstrap_descriptive": cfg["power"]["seed"]}
        run.save_json("prepare.json", {"index": idx, "projector_checks": pchk, "J_ids_sha256": ids_sha(J),
                                       "seeds": seeds, "protected_manifest_sha256": sha256_file(PROTECTED)})
        return idx
    return _staged("prepare", cfg, go)


# ================================================================ model context

class Ctx:
    def __init__(self, cfg, run, device=None, dtype=None):
        self.cfg, self.run = cfg, run
        c = copy.deepcopy(cfg)
        if dtype:
            c["model"]["dtype"] = dtype
        self.device = get_device(device or cfg["model"]["device"])
        torch.set_num_threads(cfg["budget"]["max_threads"])
        with run.timer(f"load_model_{self.device.type}_{c['model']['dtype']}"):
            self.model, self.tok = load_model(c, self.device)
        info = model_file_info(cfg)
        run.manifest.setdefault("model_files", info)
        self.P, self.pchk = load_projector(cfg, model_sha=info["safetensors_sha256"])
        self.J = year_token_ids(self.tok)
        if ids_sha(self.J) != cfg["observable"]["J_ids_sha256"]:
            raise ValueError("J token list mismatch")
        mlp = self.model.transformer.h[cfg["layer"]].mlp
        self.Wc = mlp.c_fc.weight.detach().cpu().to(F64)
        self.b_fc = mlp.c_fc.bias.detach().cpu().to(F64)
        self.pad = self.tok.eos_token_id
        self.layer = cfg["layer"]

    def batch(self, id_lists):
        return make_batch(id_lists, self.pad, self.device)

    def extract(self, id_lists, check=None, every=48):
        """CANONICAL extraction: one unpatched forward per prompt with `rows` identical copies (no padding, no
        mixing); returns the reference row's branch input (float64 CPU) and raw J logits. `check()` is called every
        `every` prompts and after the last one; a non-None status stops before the next forward (ResourceStop)."""
        rows, ref = extraction_layout(self.cfg)
        xs, lj = [], []
        with branch_mlp(self.model, self.layer) as w:
            for i, ids in enumerate(id_lists):
                b = self.batch([ids] * rows)
                check_canonical(b, ids, rows)
                o = forward(self.model, w, b)
                xs.append(o["x_in_last"][ref].cpu().to(F64))
                lj.append(o["logits"][ref, self.J].cpu().to(F64))
                if check is not None and ((i + 1) % every == 0 or i + 1 == len(id_lists)):
                    st = check()
                    if st:
                        raise ResourceStop(st, (torch.stack(xs), torch.stack(lj)))
        self.counts = getattr(self, "counts", {"extraction_forwards": 0, "extraction_rows": 0})
        self.counts["extraction_forwards"] += len(id_lists)
        self.counts["extraction_rows"] += len(id_lists) * rows
        return torch.stack(xs), torch.stack(lj)

    def group_rows(self, rec_ids, branches):
        """One forward of the canonical layout (19 copies of the recipient) with the given branch inputs."""
        rows, _ = extraction_layout(self.cfg)
        if len(branches) != rows:
            raise AssertionError("group forward must use the canonical row count")
        b = self.batch([rec_ids] * len(branches))
        check_canonical(b, rec_ids, rows)
        dtype = next(self.model.parameters()).dtype  # float64 is cast on CPU before any move to MPS
        with branch_mlp(self.model, self.layer) as w:
            o = forward(self.model, w, b, branch=branches.to(dtype).to(self.device))
        return o["logits"][:, self.J].cpu().to(F64), o


def group_branches(P, x_rec, x_ret, x_don):
    x_minus, x_plus = hybrids(P, x_rec, x_don, x_ret)
    rows = torch.cat([x_minus, x_plus, x_ret[None], x_rec[None], x_minus[:1]])
    return rows, x_minus, x_plus


def group_stats(lj):
    """From raw J logits of the 19 rows: R, U (two ways), natural response, numerical controls."""
    F = lj - lj.mean(-1, keepdim=True)
    R = F[8:16] - F[0:8]
    u1, u2 = u_variance(R), u_pairs(R)
    return {"R": R, "U": u1, "U_pairs": u2, "R_nat": F[16] - F[17], "F_nat_r": F[17], "F_dup": F[18], "F_minus0": F[0]}


# ================================================================ smoke (old calibration only)

def _mem_gib(dev):
    m = memory_snapshot(dev)
    return max(m["max_rss_bytes"], m.get("mps_driver_allocated_bytes", 0)) / 2 ** 30, m


def smoke(cfg):
    sc = cfg["smoke"]

    def go(run):
        ctx = Ctx(cfg, run)
        pairs = read_csv(ROOT / "manifests" / "pilot_pairs_calibration.csv")
        pool = read_csv(ROOT / "manifests" / "moment_pool_calibration.csv")
        rng = np.random.default_rng(sc["seed"])
        # pair 0 is the warmup prompt, so it is never a smoke group; recipients alternate between the two
        # old-calibration lengths (12, 14), and same-year donors of both lengths occur, so prompts that were padded
        # in the v2 mixed extraction are covered
        by_len = {}
        for i in range(1, len(pairs)):
            by_len.setdefault(len(pairs[i]["rec_ids"]), []).append(i)
        lens = sorted(by_len)
        pick = sorted(int(by_len[lens[k % len(lens)]][rng.integers(len(by_len[lens[k % len(lens)]]))])
                      for k in range(sc["n_groups"]))
        groups = []
        for i in pick:
            p = pairs[i]
            same = [q for q in pool if q["yy"] == p["r"]]
            don = [same[j] for j in rng.permutation(len(same))[:8]]
            groups.append({"rec": p["rec_ids"], "ret": p["don_ids"], "don": [q["ids"] for q in don], "id": p["pair_id"]})
        res = {"data": sc["data"], "groups": [g["id"] for g in groups], "projector_checks": ctx.pchk}
        all_ids = [x for g in groups for x in [g["rec"], g["ret"], *g["don"]]]
        res["prompt_lengths"] = sorted({len(x) for x in all_ids})
        if len(res["prompt_lengths"]) < 2:
            raise RuntimeError("smoke must cover prompts of different lengths")
        # untimed warmup: one extraction-shaped and one group-shaped (19-row) forward on an old calibration prompt
        # that is not one of the smoke groups; then exactly `timed_groups` groups are timed
        warmup(ctx, cfg)
        t0 = time.perf_counter()
        ids = [x for g in groups for x in [g["rec"], g["ret"], *g["don"]]]
        X, LJ = ctx.extract(ids)
        sync(ctx.device)
        t_extract = time.perf_counter() - t0
        gt, stats, unp_lj = [], [], []
        for gi, g in enumerate(groups):
            base = gi * 10
            x_rec, x_ret, x_don = X[base], X[base + 1], X[base + 2:base + 10]
            rows, x_minus, x_plus = group_branches(ctx.P, x_rec, x_ret, x_don)
            if gi == 0:
                rows0 = rows
            sync(ctx.device)
            t1 = time.perf_counter()
            lj, o = ctx.group_rows(g["rec"], rows)
            sync(ctx.device)
            gt.append(time.perf_counter() - t1)
            st = group_stats(lj)
            st["identity_max_abs"] = float((lj[17] - LJ[base]).abs().max())
            stats.append(st)
            unp_lj.append(LJ[base])
        res["u_two_way_rel"] = max(abs(s["U"] - s["U_pairs"]) / max(s["U"], 1e-300) for s in stats)
        res["identity_max_abs"] = max(s["identity_max_abs"] for s in stats)
        s2 = scale_s2(torch.stack([s["R_nat"] for s in stats]), np.full(len(stats), 1 / len(stats)))
        res["s2_oldcal_technical"] = s2
        res["U_oldcal_technical"] = [s["U"] for s in stats]

        # numerical controls on group 0
        g, base = groups[0], 0
        x_rec, x_ret, x_don = X[0], X[1], X[2:10]
        own_minus, own_plus = hybrids(ctx.P, x_rec, x_rec[None].expand(1, -1), x_ret)
        res["own_q_algebra_max_abs"] = float((own_minus[0] - x_rec).abs().max())
        same_q = x_don[:1].expand(8, -1)
        rows_sq, _, _ = group_branches(ctx.P, x_rec, x_ret, same_q)
        lj_sq, _ = ctx.group_rows(g["rec"], rows_sq)
        res["same_q_U"] = group_stats(lj_sq)["U"]
        rows_d0, _, _ = group_branches(ctx.P, x_rec, x_rec, x_don)  # x_a = x_rec -> d = 0
        lj_d0, _ = ctx.group_rows(g["rec"], rows_d0)
        res["d0_R_max_abs"] = float(group_stats(lj_d0)["R"].abs().max())
        # positions and residual bypass: earlier positions and the ln_2 input at the last position are unchanged
        b1 = ctx.batch([g["rec"]] * 2)
        cap = []
        hk = ctx.model.transformer.h[ctx.layer].ln_2.register_forward_pre_hook(lambda m, i: cap.append(i[0].detach().clone()))
        try:
            with branch_mlp(ctx.model, ctx.layer) as w:
                o_base = forward(ctx.model, w, b1)
                # group 0's own scientific branches: x_minus_0 and x_plus_0
                o_pat = forward(ctx.model, w, b1, branch=rows0[[0, 8]].to(torch.float32).to(ctx.device))
        finally:
            hk.remove()
        li = b1.last_idx
        pos = torch.arange(o_base["hidden_all"].shape[1], device=ctx.device)[None, :] < li[:, None]
        res["other_positions_max_abs"] = float((o_base["hidden_all"] - o_pat["hidden_all"]).abs()[pos].max())
        res["resid_mid_max_abs"] = float((cap[0] - cap[1]).abs().max())
        res["patched_logits_differ"] = float((o_base["logits"] - o_pat["logits"]).abs().max())
        # zero-control noise relative to the planning margin (L^2 units, with the old-calibration technical s^2)
        noise = max(res["same_q_U"], float(norm2_m(stats[0]["F_dup"] - stats[0]["F_minus0"])),
                    float(norm2_m(stats[0]["F_nat_r"] - (unp_lj[0] - unp_lj[0].mean()))))
        res["zero_noise_L2_units"] = noise / s2
        res["zero_noise_over_theta0"] = noise / s2 / cfg["power"]["theta0"]
        # CPU float64 copy for the first group(s)
        ctx64 = Ctx(cfg, run, device="cpu", dtype="float64")
        rel_R, rel_U = [], []
        for gi in range(sc["n_f64_groups"]):
            base = gi * 10
            X64, _ = ctx64.extract([groups[gi]["rec"], groups[gi]["ret"], *groups[gi]["don"]])
            rows64, _, _ = group_branches(ctx64.P, X64[0], X64[1], X64[2:10])
            lj64, _ = ctx64.group_rows(groups[gi]["rec"], rows64)
            s64 = group_stats(lj64)
            rel_R.append(float((stats[gi]["R"] - s64["R"]).norm() / s64["R"].norm()))
            rel_U.append(abs(stats[gi]["U"] - s64["U"]) / s64["U"])
            res.setdefault("f64_activation_rel", []).append(float((X[base:base + 10] - X64).norm() / X64.norm()))
        res["f32_vs_f64_R_rel"], res["f32_vs_f64_U_rel"] = max(rel_R), max(rel_U)
        del ctx64
        # legacy (v2-style) mixed, right-padded extraction of the SAME old-calibration smoke prompts, with extra pad
        # positions P: descriptive contrast showing the layout effect the canonical path removes (not a gate)
        res["legacy_mixed_padded_contrast_descriptive"] = legacy_contrast(ctx, groups, sc["legacy_pads"])
        # timing and projection for the 48 development groups (scientific + scale + control rows; extraction)
        n = cfg["design"]["n_groups"]
        per_prompt = t_extract / len(ids)
        res["timing"] = {"warmup": "untimed: one extraction forward + one 19-row group forward on an old calibration prompt",
                         "timed_groups": len(gt), "extract_s": t_extract, "extract_per_prompt_s": per_prompt, "group_forward_s": gt,
                         "group_forward_max_s": max(gt),
                         "projected_dev_model_stage_s": (max(gt) * n + per_prompt * n * 10) * cfg["budget"]["projection_overhead_factor"],
                         "rows_per_group": len(ROW_KINDS), "scientific_forwards_dev": 16 * n,
                         "scale_forwards_dev": 2 * n, "control_forwards_dev": n, "extraction_forwards_dev": 10 * n,
                         "extraction_rows_dev": 10 * n * len(ROW_KINDS),
                         "note": "extraction is one canonical 19-row forward per logical prompt"}
        res["canonical_layout_counts"] = dict(ctx.counts)
        gib, snap = _mem_gib(ctx.device)
        res["memory"] = {**snap, "peak_gib": gib}
        gt_ = sc["gates"]
        checks = {"canonical_layout": ctx.counts["extraction_rows"] == ctx.counts["extraction_forwards"] * len(ROW_KINDS),
                  "identity": res["identity_max_abs"] <= gt_["identity_max_abs"],
                  "own_q_algebra": res["own_q_algebra_max_abs"] <= gt_["own_q_algebra_max_abs"],
                  "same_q": res["same_q_U"] <= gt_["same_q_U_max"],
                  "d0": res["d0_R_max_abs"] <= gt_["d0_R_max_abs"],
                  "other_positions": res["other_positions_max_abs"] <= gt_["other_positions_max_abs"],
                  "resid_mid": res["resid_mid_max_abs"] <= gt_["resid_mid_max_abs"],
                  "f32_vs_f64_R": res["f32_vs_f64_R_rel"] <= gt_["f32_vs_f64_R_rel"],
                  "f32_vs_f64_U": res["f32_vs_f64_U_rel"] <= gt_["f32_vs_f64_U_rel"],
                  "u_two_way": res["u_two_way_rel"] <= gt_["u_two_way_rel"],
                  "zero_noise": res["zero_noise_over_theta0"] <= gt_["zero_noise_over_theta0_max"],
                  "patch_has_effect": res["patched_logits_differ"] > 0,
                  "budget_projection": res["timing"]["projected_dev_model_stage_s"] <= cfg["budget"]["model_stage_cap_s"]
                  and gib <= cfg["budget"]["memory_cap_gib"]}
        res["gates"] = {k: bool(v) for k, v in checks.items()}
        res["all_gates_pass"] = all(checks.values())
        run.save_json("smoke.json", res)
        if not res["all_gates_pass"]:
            raise RuntimeError(f"smoke gates failed: {[k for k, v in checks.items() if not v]}")
        return res
    return _staged("smoke", cfg, go)


# ================================================================ registration

def reg_path(cfg):
    return REGISTRATION if REGISTRATION is not None else registration_path(cfg)


def register(cfg, smoke_run, tests_log):
    rp = reg_path(cfg)
    if rp.exists():
        raise FileExistsError(f"{rp} exists; never overwritten")
    if cfg["protocol_version"].endswith("DRAFT"):
        raise RuntimeError("finalise protocol_version before registration")
    d, man = load_run(smoke_run, "smoke")
    sm = json.loads((d / "smoke.json").read_text())
    if not sm["all_gates_pass"]:
        raise RuntimeError("smoke gates did not pass")
    cur = bound_hashes(cfg)
    if man.get("config_path") != cur["config_path"] or man["bound"].get("manifest_dir") != cur["manifest_dir"]:
        raise RuntimeError("smoke run used a different config file or manifest directory")
    if cur["scp_dependency_tree_sha256"] != cfg["scp_dependency_tree_sha256"]:
        raise RuntimeError("scp dependency tree differs from the config")
    for k in ("source", "scp_dependency_tree_sha256", "manifest_dir", "manifests", "protected_manifest_sha256",
              "protocol_path", "protocol_sha256", "artifact_file_sha256", "J_ids_sha256"):
        if man["bound"][k] != cur[k]:
            raise RuntimeError(f"{k} changed since smoke run {smoke_run}")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("protocol_version", "_path")}  # noqa: E731
    smoke_cfg = json.loads((d / "snapshot" / cur["config_path"]).read_text())
    if strip(smoke_cfg) != strip(cfg):
        raise RuntimeError("config differs from the smoke run beyond the protocol_version label")
    vp = verify_protected()
    if not vp["ok"]:
        raise RuntimeError(f"protected files changed: {vp}")
    v2 = verify_v2_preservation(cfg)
    if not v2["ok"]:
        raise RuntimeError(f"registered v2 bytes changed: {v2['problems']}")
    if not cur["manifests"]:
        raise RuntimeError("no manifests; run prepare first")
    reg = {"created_utc": utcnow().isoformat(), "protocol_version": cfg["protocol_version"], "bound": cur,
           "smoke_run": smoke_run, "smoke_json_sha256": sha256_file(d / "smoke.json"), "tests_log": tests_log,
           "tests_log_sha256": sha256_file(ROOT / tests_log) if tests_log else None,
           "v2_preservation": cfg["v2_preservation"], "v2_status": "EVALUATION_INVALID (preserved, not reclassified)",
           "rule": "technical remeasurement of the registered v2 design: no v3 development outcome exists before this "
                   "file; v2 raw outputs exist (the remeasurement is not blind)"}
    write_once(rp, reg)
    return reg


def verify_registration(cfg):
    rp = reg_path(cfg)
    if not rp.exists():
        raise RuntimeError("no development registration; run `register` first")
    if cfg["protocol_version"].endswith("DRAFT"):
        raise RuntimeError("DRAFT protocol cannot run past registration")
    reg = json.loads(rp.read_text())
    if reg.get("protocol_version") != cfg["protocol_version"]:
        raise RuntimeError("registration belongs to another protocol version")
    v2 = verify_v2_preservation(cfg)
    if not v2["ok"]:
        raise RuntimeError(f"registered v2 bytes changed: {v2['problems']}")
    cur = bound_hashes(cfg)
    bad = [k for k in cur if cur[k] != reg["bound"][k]]
    if bad:
        raise RuntimeError(f"bound files changed since registration: {bad}")
    return reg


def legacy_batch(ctx, id_lists, extra_pad):
    """v2-style layout: mixed prompts, right-padded to the longest, plus `extra_pad` masked positions."""
    b = make_batch([x + [ctx.pad] * extra_pad for x in id_lists], ctx.pad, ctx.device)
    for i, x in enumerate(id_lists):
        b.mask[i, len(x):] = 0
    b.last_idx[:] = torch.tensor([len(x) - 1 for x in id_lists], device=ctx.device)
    return b


def legacy_contrast(ctx, groups, pads):
    """For each extra pad P: the v2-style identity control (group forward, branch = x from the mixed padded
    extraction, row 17, vs the mixed padded extraction's own logits), per smoke recipient. Descriptive."""
    ids = [x for g in groups for x in [g["rec"], g["ret"], *g["don"]]]
    out = {}
    for P in pads:
        b = legacy_batch(ctx, ids, P)
        with branch_mlp(ctx.model, ctx.layer) as w:
            o = forward(ctx.model, w, b)
        xl, ll = o["x_in_last"].cpu().to(F64), o["logits"][:, ctx.J].cpu().to(F64)
        diffs = []
        for gi, g in enumerate(groups):
            base = gi * 10
            lj, _ = ctx.group_rows(g["rec"], xl[base].expand(len(ROW_KINDS), -1))
            diffs.append({"recipient_len": len(g["rec"]), "pad": int(b.ids.shape[1] - len(g["rec"])),
                          "identity_max_abs": float((lj[ROW_KINDS.index("nat_r")] - ll[base]).abs().max())})
        out[f"extra_pad{P}"] = diffs
    return out


# ================================================================ development (48 groups)

def warmup(ctx, cfg):
    """Untimed warmup on an OLD calibration prompt (no development outcome): one extraction-shaped forward and one
    19-row group-shaped forward whose branches are the prompt's own input (identity)."""
    rec = read_csv(ROOT / "manifests" / "pilot_pairs_calibration.csv")[0]["rec_ids"]
    x, _ = ctx.extract([rec])
    ctx.group_rows(rec, x.expand(len(ROW_KINDS), -1))
    sync(ctx.device)


def resource_status(elapsed, gib, b, projected=None):
    """Hard caps (checked after extraction and after every group) and the projection (after `timed_groups`)."""
    if elapsed > b["model_stage_cap_s"]:
        return "resource_stop_time_cap"
    if gib > b["memory_cap_gib"]:
        return "resource_stop_memory_cap"
    if projected is not None and projected > b["model_stage_cap_s"]:
        return "resource_stop_projection"
    return None


def run_groups(ctx, groups, X, LJ, b, clock, memory):
    """Group loop with a resource check after every group. A stop happens before the next forward (an in-flight
    forward cannot be pre-empted). Returns partial or complete lists and a status."""
    raw, stats, sup_rows, ctl_rows, timing = [], [], [], [], []
    status, budget = "ok", {}
    t_start = clock.start
    for gi, g in enumerate(groups):
        base = gi * 10
        x_rec, x_ret, x_don = X[base], X[base + 1], X[base + 2:base + 10]
        rows, x_minus, x_plus = group_branches(ctx.P, x_rec, x_ret, x_don)
        t1 = clock()
        lj, _ = ctx.group_rows(g["rec_ids"], rows)
        timing.append(clock() - t1)
        raw.append(lj)
        stats.append(group_stats(lj))
        ctl_rows.append({"group_id": g["group_id"], "identity_max_abs": float((lj[17] - LJ[base]).abs().max()),
                         "duplicate_max_abs": float((lj[18] - lj[0]).abs().max()),
                         "finite": bool(torch.isfinite(lj).all())})
        sp = support(ctx.P, x_rec, x_don, x_ret, x_minus, x_plus, ctx.Wc, ctx.b_fc)
        for k in range(8):
            sup_rows.append({"group_id": g["group_id"], "slot": k, **{n: float(v[k]) for n, v in sp.items()}})
        elapsed, gib = clock() - t_start, memory()
        projected = None
        if gi + 1 == b["timed_groups"]:
            projected = elapsed + max(timing) * (len(groups) - gi - 1) * b["projection_overhead_factor"]
            budget = {"elapsed_s": elapsed, "group_times_s": list(timing), "projected_model_stage_s": projected,
                      "peak_gib": gib}
        st = resource_status(elapsed, gib, b, projected)
        if st and gi + 1 < len(groups):
            status = st
            budget.update(stopped_after_group=gi + 1, stop_elapsed_s=elapsed, stop_gib=gib)
            break
        if st:  # after the final group there is no further forward to stop; recorded, not hidden
            budget.update(cap_exceeded_after_final_group=st, final_elapsed_s=elapsed, final_gib=gib)
    return {"raw": raw, "stats": stats, "support": sup_rows, "controls": ctl_rows, "timing": timing,
            "status": status, "budget": budget}


class _Clock:
    def __init__(self, sync_fn):
        self.sync_fn = sync_fn
        self.start = self()

    def __call__(self):
        self.sync_fn()
        return time.perf_counter()


def _save_partial(run, ctx, groups, res):
    if res["raw"]:
        torch.save({"row_kinds": ROW_KINDS, "group_ids": [g["group_id"] for g in groups[:len(res["raw"])]],
                    "J_ids": ctx.J, "raw_J_logits": torch.stack(res["raw"]),
                    "R_centred": torch.stack([s["R"] for s in res["stats"]]),
                    "R_nat_centred": torch.stack([s["R_nat"] for s in res["stats"]])}, run.dir / "raw_J_logits.pt")
        write_csv(run.dir / "controls.csv", res["controls"], mode="w")
        write_csv(run.dir / "support.csv", res["support"], mode="w")


def develop(cfg):
    reg = verify_registration(cfg)
    for p in RUNS.glob("*_gate_develop"):
        m = json.loads((p / "run_manifest.json").read_text())
        if m.get("status") == "ok" and m.get("protocol_version") == cfg["protocol_version"]:
            raise FileExistsError("an ok development run already exists for this protocol version; it is run once")
    b = cfg["budget"]

    def go(run):
        run.manifest["first_outcome_utc_not_before"] = utcnow().isoformat()
        run.write_manifest()
        groups, _ = load_groups(cfg)
        ctx = Ctx(cfg, run)  # model load is excluded from the model-stage clock
        warmup(ctx, cfg)     # warmup is excluded too (old calibration prompt only)
        clock = _Clock(lambda: sync(ctx.device))
        memory = lambda: _mem_gib(ctx.device)[0]  # noqa: E731
        ids = [x for g in groups for x in [g["rec_ids"], g["ret_ids"], *[g[f"don{k}_ids"] for k in range(8)]]]
        order = [f"{g['group_id']}-{r}" for g in groups for r in ["rec", "ret", *[f"d{k}" for k in range(8)]]]
        prov = {"layout": cfg["extraction"]["layout"], "rows": cfg["extraction"]["rows"],
                "reference_row": cfg["extraction"]["reference_row"],
                "rule": "each prompt alone, replicated, unpadded; x and the unpatched reference from the reference row"}
        try:
            with run.timer("extract"):
                X, LJ = ctx.extract(ids, check=lambda: resource_status(clock() - clock.start, memory(), b),
                                    every=b["extraction_check_every"])
        except ResourceStop as e:  # stop before the next forward; keep the partial activations
            torch.save({"prompt_order": order[:len(e.done[0])], "X": e.done[0], "unpatched_J_logits": e.done[1],
                        "provenance": prov}, run.dir / "activations.pt")
            run.manifest["budget"] = {"stopped_during": "extraction", "status": e.status, "n_extracted": len(e.done[0])}
            run.save_json("development.json", {"status": e.status, "stopped_during": "extraction"})
            raise RuntimeError(f"{e.status} during extraction; stopped before the next forward (partial activations kept)")
        torch.save({"prompt_order": order, "X": X, "unpatched_J_logits": LJ, "provenance": prov},
                   run.dir / "activations.pt")
        run.manifest["compute_counts"] = {"logical_prompts_extracted": len(ids), **ctx.counts,
                                          "group_forwards": len(groups), "group_rows": len(groups) * len(ROW_KINDS),
                                          "scientific_rows": 16 * len(groups), "scale_rows": 2 * len(groups),
                                          "control_rows": len(groups)}
        run.write_manifest()
        res = run_groups(ctx, groups, X, LJ, b, clock, memory)
        _save_partial(run, ctx, groups, res)
        gib, snap = _mem_gib(ctx.device)
        run.manifest.update(model_stage_s=clock() - clock.start, memory={**snap, "peak_gib": gib},
                            budget=res["budget"])
        if res["status"] != "ok":
            run.save_json("development.json", {"status": res["status"], "budget": res["budget"],
                                               "n_groups_done": len(res["raw"])})
            raise RuntimeError(f"{res['status']}; stopped before the next group (partial raw outputs and controls kept)")
        stats = res["stats"]
        out = summarise(cfg, groups, stats, res["controls"])
        write_csv(run.dir / "groups_U.csv",
                  [{"group_id": g["group_id"], "frame": g["frame"], "template": g["template"],
                    "direction": g["direction"], "gap_bin": g["gap_bin"], "r": g["r"], "a": g["a"],
                    "U": s["U"], "U_pairs": s["U_pairs"], "R_nat_norm2_m": float(norm2_m(s["R_nat"]))}
                   for g, s in zip(groups, stats)], mode="w")
        out["timing"] = {"group_forward_s": res["timing"], "model_stage_s": run.manifest["model_stage_s"],
                         "clock": "starts after model load and warmup; covers extraction and all group forwards"}
        out["label"] = ("development TECHNICAL REMEASUREMENT (v3, canonical extraction layout) of the registered v2 "
                        "design; v2 remains EVALUATION_INVALID; v2 raw outputs (s2, U ranges) had been produced, so "
                        "this remeasurement is not blind or unexposed; not confirmatory")
        run.save_json("development.json", out)
        if not out["valid"]:
            raise RuntimeError(f"development measurement invalid: {out['invalid_reasons']}")
        return out
    return _staged("develop", cfg, go, {"registration_sha256": sha256_file(reg_path(cfg)), "reg_created": reg["created_utc"]})


def summarise(cfg, groups, stats, ctl_rows):
    num = cfg["numerics"]
    n = len(groups)
    w = np.full(n, 1.0 / n)
    U = np.array([s["U"] for s in stats])
    U2 = np.array([s["U_pairs"] for s in stats])
    s2 = scale_s2(torch.stack([s["R_nat"] for s in stats]), w)
    noise = max(max(c["identity_max_abs"] for c in ctl_rows), max(c["duplicate_max_abs"] for c in ctl_rows))
    noise_sq = max(float(norm2_m(s["F_dup"] - s["F_minus0"])) for s in stats)
    reasons = []
    if not all(c["finite"] for c in ctl_rows) or not np.isfinite(U).all() or not np.isfinite(s2):
        reasons.append("non-finite output")
    if np.max(np.abs(U - U2) / np.maximum(U, 1e-300)) > num["u_two_way_rel_tol"]:
        reasons.append("U two-way mismatch")
    if max(c["identity_max_abs"] for c in ctl_rows) > num["identity_max_abs"]:
        reasons.append("identity replacement differs")
    if max(c["duplicate_max_abs"] for c in ctl_rows) > num["duplicate_max_abs"]:
        reasons.append("duplicate rows differ")
    if not (s2 > num["s2_abs_floor"] and s2 >= num["s2_over_noise_min"] * noise_sq):
        reasons.append("scale s^2 not distinguishable from zero/noise")
    valid = not reasons
    lhat2 = float(w @ U) / s2 if valid else None
    T = template_T(groups, U, s2) if valid else None
    m, v, mbar = frame_moments(T) if valid else (None, None, None)
    out = {"valid": valid, "invalid_reasons": reasons, "n_groups": n, "s2": s2, "s": s2 ** 0.5,
           "Lhat2": lhat2, "Lhat": lhat2 ** 0.5 if valid else None, "template_T": T, "frame_mean_T": m,
           "frame_var_T": v, "mbar": mbar, "U_min": float(U.min()), "U_max": float(U.max()),
           "numerical_noise_max_abs": noise, "numerical_noise_norm2_m": noise_sq,
           "noise_T": noise_sq / s2 if s2 > 0 else None, "u_two_way_max_rel": float(np.max(np.abs(U - U2) / np.maximum(U, 1e-300))),
           "label": "development result only; not confirmatory"}
    if valid:
        bs = template_bootstrap(T, cfg["power"]["bootstrap_B"], cfg["power"]["seed"], cfg["power"]["one_sided_conf"])
        out["dev_bootstrap_descriptive"] = {**bs, "decision_if_this_were_confirmatory_NOT_APPLICABLE":
                                            confirm_decision(bs, cfg["confirmatory_rule_draft"]["L0"])}
    return out


# ================================================================ power planning

def power(cfg, dev_run):
    reg = verify_registration(cfg)
    d, man = load_run(dev_run, "develop")
    if man["bound"] != reg["bound"]:
        raise RuntimeError("development run bound hashes differ from the registration")
    dev = json.loads((d / "development.json").read_text())

    def go(run):
        if not dev["valid"]:
            res = {"decision": "EVALUATION_INVALID", "reason": dev["invalid_reasons"]}
        else:
            res = run_power(cfg, dev["frame_mean_T"], dev["frame_var_T"], dev["mbar"], dev["noise_T"])
        res["dev_run"] = dev_run
        res["assumptions"] = ("Gamma shape, the fourfold variance sensitivity and transfer to new frames are "
                              "assumptions, not guarantees; s is fixed after development and the plan is conditional on it")
        run.save_json("power.json", res)
        if res.get("rows"):
            flat = []
            for r in res["rows"]:
                for h in ("null", "alt"):
                    flat.append({"N": r["N"], "templates_per_frame": r["templates_per_frame"], "scenario": r["scenario"],
                                 "hypothesis": h, **r[h]})
            write_csv(run.dir / "power_rows.csv", flat, mode="w")
        return res
    return _staged("power", cfg, go, {"dev_run": dev_run, "registration_sha256": sha256_file(reg_path(cfg))})


def report(cfg, dev_run, power_run=None):
    """Re-aggregate the development run from raw files with the separate implementation and collect numbers."""
    from .aggregate import aggregate
    d, _ = load_run(dev_run, "develop")

    def go(run):
        agg = aggregate(d)
        dev = json.loads((d / "development.json").read_text())
        diff = {"Lhat2_abs_diff": abs(agg["Lhat2"] - dev["Lhat2"]), "s2_abs_diff": abs(agg["s2"] - dev["s2"]),
                "U_max_abs_diff": agg["U_max_abs_diff_vs_groups_U_csv"]}
        out = {"dev_run": dev_run, "separate_aggregation": agg, "agreement": diff, "development": dev,
               "protected_files": verify_protected()}
        if power_run:
            pd_, _ = load_run(power_run, "power")
            out["power"] = json.loads((pd_ / "power.json").read_text())
        run.save_json("report_numbers.json", out)
        return out
    return _staged("report", cfg, go, {"dev_run": dev_run, "power_run": power_run})
