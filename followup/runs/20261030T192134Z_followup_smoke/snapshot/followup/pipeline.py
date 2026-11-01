"""Follow-up stages: prepare | smoke | register | fit | validate | freeze | predict-test | evaluate-test | report.

Order and locks (followup/PROTOCOL_FOLLOWUP.md §7):
* registration binds protocol/design/config/source/manifests before any fit/validation outcome;
* freeze binds the ridge selection, s and the validation gate before any test forward;
* predict-test computes test features/predictions from unperturbed forwards only (labels locked) and writes a
  write-once lock with their hashes;
* evaluate-test verifies that lock before the first perturbed test forward (the labels).
"""
import copy
import csv
import json
import time

import numpy as np
import torch

from scp.common import get_device, load_config as load_cfg3, memory_snapshot, set_seed, sha256_file, sync, utcnow
from scp.data import make_batch, read_csv
from scp.day3 import load_artifact
from scp.metrics import kl
from scp.model import forward_last, load_model, make_reduction, model_file_info, wrapped_mlp

from . import ROOT
from .common import FROZEN, REGISTRATION, TEST_LOCK, FRun, bound_hashes, load_run, tensor_sha, write_once
from .data import load_split, prepare as prepare_manifests, write_csv
from .downstream import centred_probes, exact_jvp, leaf_logits, probe_vjps, rademacher_probes
from .features import design_matrix, feature_rows, label_norm, response_norm, scale_from_fit, shuffle_perm
from .local import LocalMLP
from .ridge import ridge_fit, ridge_predict, select_lambda
from .stats import decision, gain, weighted_mse

F64 = torch.float64
TEST_EVALUATED = FROZEN.parent / "FOLLOWUP_TEST_EVALUATED.json"


class LabelLock(RuntimeError):
    pass


# ================================================================ context

class Ctx:
    """Model, frozen Day-3 PCA192+bias artifact, float64 local MLP, fixed probes and permutation."""

    def __init__(self, cfg, run, labels_allowed):
        self.cfg, self.run, self.labels_allowed = cfg, run, labels_allowed
        self.layer = cfg["layer"]
        self.device = get_device(cfg["model"]["device"])
        set_seed(cfg["seed"])
        with run.timer("load_model"):
            self.model, self.tok = load_model(cfg, self.device)
        info = model_file_info(cfg)
        run.manifest["model"].update(info)
        self.pad = self.tok.eos_token_id
        self.mlp = self.model.transformer.h[self.layer].mlp
        a = cfg["artifact"]
        cfg3 = load_cfg3(ROOT / a["day3_config"])
        bases, moments, _ = load_artifact(cfg3, expected_file_sha=a["file_sha256"],
                                          model_sha=info["safetensors_sha256"])
        basis = bases[(a["basis"], a["k"])]
        mu, sigma = moments[(a["basis"], a["k"])][a["moment_pool"]]
        self.local = LocalMLP.from_mlp(self.mlp, basis.P, mu)
        self.m_B, self.m_B_residual = self.local.discarded_mean_input()
        if self.m_B_residual > 1e-8:
            raise ValueError(f"discarded mean input not recovered from mu (rel residual {self.m_B_residual:.2e})")
        self.red = make_reduction(basis.row_maps()[0], self.mlp, mu, sigma, a["method"], a["scope"],
                                  cfg3["quadrature_nodes"], cfg3["quadrature_sigma_max_checked"], self.device)
        p = cfg["probes"]
        self.U = rademacher_probes(p["n"], self.model.config.vocab_size, p["seed"])
        self.Uc_dev = centred_probes(self.U).to(torch.float32).to(self.device)
        self.perm = shuffle_perm(self.local.P.shape[0], cfg["shuffle"]["seed"])
        run.manifest["fixed"] = {"probes_sha256": tensor_sha(self.U), "perm_sha256": tensor_sha(self.perm),
                                 "artifact_P_sha256": tensor_sha(basis.P), "artifact_mu_sha256": tensor_sha(mu),
                                 "m_B_rel_residual": self.m_B_residual}
        run.write_manifest()

    def batches(self, rows, bs):
        for s in range(0, len(rows), bs):
            part = rows[s:s + bs]
            yield s, (make_batch([p["rec_ids"] for p in part], self.pad, self.device),
                      make_batch([p["don_ids"] for p in part], self.pad, self.device))

    def unperturbed(self, rows, with_vjp=True):
        """Original/reduced forwards at the UNPERTURBED recipient and the donor, plus probe VJPs.
        No intervention is applied here."""
        out = {k: [] for k in ("x_r", "x_a", "orig_b", "red_b", "G", "m0_check", "leaf_diff")}
        with wrapped_mlp(self.model, self.layer) as w:
            for _, (br, ba) in self.batches(rows, self.cfg["budget"]["batch_size"]):
                o_r = forward_last(self.model, w, br)
                o_a = forward_last(self.model, w, ba)
                o_red = forward_last(self.model, w, br, reduction=self.red)
                out["x_r"].append(o_r["x_pre_last"].cpu().to(F64))
                out["x_a"].append(o_a["x_pre_last"].cpu().to(F64))
                out["orig_b"].append(o_r["logits"].cpu())
                out["red_b"].append(o_red["logits"].cpu())
                if with_vjp:
                    G, m0, logits0, ld = self._vjp_outside(w, br)
                    out["G"].append(G.cpu().to(F64))
                    out["m0_check"].append(float((m0 - o_r["out_last"]).abs().max()))
                    out["leaf_diff"].append(max(ld, float((logits0 - o_r["logits"]).abs().max())))
        res = {k: torch.cat(v) for k, v in out.items() if k in ("x_r", "x_a", "orig_b", "red_b", "G") and v}
        res["kl"] = kl(res["orig_b"], res["red_b"])
        res["m0_max_abs"] = max(out["m0_check"], default=None)
        res["leaf_logits_max_abs"] = max(out["leaf_diff"], default=None)
        return res

    def _vjp_outside(self, w, br):
        """Temporarily restore the original MLP so the leaf wrapper can be installed (no nesting)."""
        block = self.model.transformer.h[self.layer]
        block.mlp = w.orig
        try:
            return probe_vjps(self.model, self.layer, br, self.Uc_dev)
        finally:
            block.mlp = w

    def labels(self, rows, unp, split):
        """Perturbed forwards x' = x_r + alpha P (x_a - x_r) (original and reduced). Locked on the test split
        until evaluate-test."""
        if not self.labels_allowed:
            raise LabelLock("perturbed forwards are locked in this stage")
        out = []
        bs = self.cfg["budget"]["label_batch_size"]
        x_err = 0.0
        with wrapped_mlp(self.model, self.layer) as w:
            for s, (br, _) in self.batches(rows, bs):
                sl = slice(s, s + len(br.ids))
                x_r, x_a = unp["x_r"][sl], unp["x_a"][sl]
                for alpha in self.cfg["intervention"]["alphas"]:
                    d64 = self.local.delta(x_r, x_a, alpha)
                    d = d64.to(torch.float32).to(self.device)
                    o_p = forward_last(self.model, w, br, delta=d)
                    r_p = forward_last(self.model, w, br, delta=d, reduction=self.red)
                    x_err = max(x_err, float((o_p["x_last"].cpu().to(F64) - (x_r + d64)).abs().max()))
                    op, rp = o_p["logits"].cpu(), r_p["logits"].cpu()
                    lab = label_norm(rp, unp["red_b"][sl], op, unp["orig_b"][sl])
                    on = response_norm(op, unp["orig_b"][sl])
                    rn = response_norm(rp, unp["red_b"][sl])
                    for i, p in enumerate(rows[sl]):
                        out.append({"rid": p["rid"], "split": split, "alpha": float(alpha),
                                    "label_raw": float(lab[i]), "orig_response_norm": float(on[i]),
                                    "red_response_norm": float(rn[i])})
        self.run.manifest.setdefault("label_checks", {})[split] = {"x_prime_max_abs_err": x_err}
        return out


def join_labels(feat_rows, label_rows, s):
    lab = {(r["rid"], r["alpha"]): r for r in label_rows}
    if len(lab) != len(feat_rows):
        raise ValueError("label/feature row mismatch")
    out = []
    for f in feat_rows:
        r = lab[(f["rid"], f["alpha"])]
        out.append({**f, "label_raw": r["label_raw"], "orig_response_norm": r["orig_response_norm"],
                    "red_response_norm": r["red_response_norm"], "Y": r["label_raw"] / s})
    return out


def model_features(cfg, name):
    return cfg["features"]["baseline"] + cfg["features"]["models"][name]


# ================================================================ registration / freeze checks

def verify_registration(cfg):
    if not REGISTRATION.exists():
        raise RuntimeError("no follow-up registration; run `register` first")
    reg = json.loads(REGISTRATION.read_text())
    cur = bound_hashes(cfg)
    bad = [k for k in cur if cur[k] != reg["bound"][k]]
    if bad:
        raise RuntimeError(f"follow-up files changed since registration: {bad}")
    if cfg["protocol_version"].endswith("DRAFT"):
        raise RuntimeError("DRAFT protocol cannot be run past registration")
    return reg


def _check_run_bound(man, reg):
    if man["bound"] != reg["bound"]:
        raise RuntimeError(f"{man['run_id']}: run was made under different bound hashes than the registration")


def load_frozen(cfg, reg):
    if not FROZEN.exists():
        raise RuntimeError("no freeze file; run `freeze` after validation")
    fr = json.loads(FROZEN.read_text())
    if fr["registration_sha256"] != sha256_file(REGISTRATION) or fr["source_tree_sha256"] != reg["bound"]["source"]["tree_sha256"]:
        raise RuntimeError("freeze file does not match the registration / current source tree")
    return fr


# ================================================================ stages

def _staged(stage, cfg, fn, extra=None):
    run = FRun(stage, cfg, extra)
    try:
        res = fn(run)
        run.finish("ok")
        return run, res
    except Exception as e:  # keep failed runs
        run.finish("failed", e)
        raise


def prepare(cfg):
    def go(run):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        idx = prepare_manifests(tok, cfg)
        run.save_json("prepare.json", idx)
        return idx
    return _staged("prepare", cfg, go)


def old_cal_rows(cfg):
    rows = read_csv(ROOT / "manifests" / "pilot_pairs_calibration.csv")
    rng = np.random.default_rng(cfg["smoke"]["seed"])
    pick = sorted(rng.choice(len(rows), size=cfg["smoke"]["n_pairs"], replace=False).tolist())
    return [{"rid": rows[i]["pair_id"], "split": "oldcal", "frame": rows[i]["frame"], "noun": rows[i]["noun"],
             "template": rows[i]["template"], "direction": rows[i]["direction"], "diff_bin": rows[i]["diff_bin"],
             "r": rows[i]["r"], "d": rows[i]["d"], "rec_ids": rows[i]["rec_ids"], "don_ids": rows[i]["don_ids"]}
            for i in pick]


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def smoke(cfg):
    """Technical smoke on OLD Day-1 calibration pairs only (never the follow-up splits): gradient FD sanity at
    two step sizes (MPS float32 and CPU float64), exact-JVP comparison, local b equivalence, self-patch,
    bypass/positions, determinism, timings and peak memory."""
    sc = cfg["smoke"]

    def go(run):
        ctx = Ctx(cfg, run, labels_allowed=True)
        rows = old_cal_rows(cfg)
        res = {"data": "manifests/pilot_pairs_calibration.csv (Day-1 calibration; technical smoke only)",
               "n_pairs": len(rows), "rids": [p["rid"] for p in rows]}
        dev = ctx.device
        sync(dev)
        t0 = time.perf_counter()
        unp = ctx.unperturbed(rows)
        sync(dev)
        t_unp = time.perf_counter() - t0
        # determinism: identical batch shapes (gate); a different batch shape only changes MPS roundoff, so it is a
        # separate descriptive check with a tolerance fixed in the config before the smoke
        unp_same = ctx.unperturbed(rows)
        res["same_shape_rerun_G_max_abs"] = float((unp_same["G"] - unp["G"]).abs().max())
        res["same_shape_rerun_logits_max_abs"] = float((unp_same["orig_b"] - unp["orig_b"]).abs().max())
        bs = cfg["budget"]["batch_size"]
        unp8 = ctx.unperturbed(rows[:bs // 2])
        cb = _rel(unp8["G"], unp["G"][:bs // 2])
        tol = sc["descriptive_tolerances"]["cross_batch_shape_G_rel"]
        res["cross_batch_shape_descriptive"] = {"batch_sizes": [bs, bs // 2], "G_rel": cb, "tolerance": tol,
                                                "within_tolerance": cb <= tol}
        res["leaf_logits_max_abs"] = unp["leaf_logits_max_abs"]
        res["leaf_m0_vs_wrapper_out_max_abs"] = unp["m0_max_abs"]
        t0 = time.perf_counter()
        labs = ctx.labels(rows, unp, "oldcal")
        sync(dev)
        t_lab = time.perf_counter() - t0
        s_cal = float(np.sqrt(np.mean([r["orig_response_norm"] ** 2 for r in labs if r["alpha"] == 1.0])))
        feats = join_labels(feature_rows(cfg, ctx.local, ctx.m_B, rows, unp, s_cal, ctx.perm), labs, s_cal)
        res["s_oldcal_technical"] = s_cal
        res["a_B_max_abs"] = max(f["a_B_max_abs"] for f in feats)
        res["x_prime_max_abs_err"] = run.manifest["label_checks"]["oldcal"]["x_prime_max_abs_err"]

        # local b equivalence (unperturbed and alpha=1), model f32 vs local f64
        n = sc["n_fd_recipients"]
        sub = rows[:n]
        _, (br, _) = next(ctx.batches(sub, n))
        x_r, x_a = unp["x_r"][:n], unp["x_a"][:n]
        d64 = ctx.local.delta(x_r, x_a, 1.0)
        with wrapped_mlp(ctx.model, ctx.layer) as w:
            oo = forward_last(ctx.model, w, br)
            orr = forward_last(ctx.model, w, br, reduction=ctx.red)
            dd = d64.float().to(dev)
            op = forward_last(ctx.model, w, br, delta=dd)
            rp = forward_last(ctx.model, w, br, delta=dd, reduction=ctx.red)
            z = torch.zeros_like(dd)
            oz = forward_last(ctx.model, w, br, delta=z)
            rz = forward_last(ctx.model, w, br, delta=z, reduction=ctx.red)
        b_model = (orr["out_last"] - oo["out_last"]).cpu().to(F64)
        bp_model = (rp["out_last"] - op["out_last"]).cpu().to(F64)
        res["local_b_rel_err_unperturbed"] = _rel(b_model, ctx.local.b(x_r))
        res["local_b_rel_err_alpha1"] = _rel(bp_model, ctx.local.b(x_r + d64))
        res["local_delta_b_rel_err_alpha1"] = _rel(bp_model - b_model, ctx.local.b(x_r + d64) - ctx.local.b(x_r))
        res["self_patch_max_abs"] = max(float((oz["logits"] - oo["logits"]).abs().max()),
                                        float((rz["logits"] - orr["logits"]).abs().max()))
        last = br.last_idx
        pos = torch.arange(op["hidden_all"].shape[1], device=dev)[None, :] < last[:, None]
        res["other_positions_max_abs"] = float((op["hidden_all"] - oo["hidden_all"]).abs()[pos].max())
        lq0 = ctx.local.quantities(x_r, x_a, 0.0, ctx.m_B)
        res["alpha0_delta_b_max_abs"] = float(lq0["delta_b"].abs().max())

        # FD sanity of the probe VJPs, MPS float32, two step sizes, random unit and delta_b directions
        g = torch.Generator().manual_seed(sc["seed"])
        v_rand = torch.randn(n, x_r.shape[1], generator=g, dtype=F64)
        v_rand = v_rand / v_rand.norm(dim=-1, keepdim=True)
        db = ctx.local.b(x_r + d64) - ctx.local.b(x_r)
        v_db = db / db.norm(dim=-1, keepdim=True)
        m0 = oo["out_last"]
        Uc64 = centred_probes(ctx.U)
        G32 = unp["G"][:n]
        fd = {}
        for vname, v in (("random", v_rand), ("delta_b", v_db)):
            an = torch.einsum("bjd,bd->bj", G32, v)
            for eps in sc["fd_eps"]:
                vp = (eps * v).float().to(dev)
                lp = leaf_logits(ctx.model, ctx.layer, br, m0 + vp).cpu().to(F64)
                lm = leaf_logits(ctx.model, ctx.layer, br, m0 - vp).cpu().to(F64)
                num = (lp - lm) @ Uc64.T / (2 * eps)
                fd[f"f32|{vname}|eps={eps}"] = max(_rel(num[i], an[i]) for i in range(n))
        res["fd_rel_err_f32"] = fd
        res["m0_norm_mean"] = float(m0.norm(dim=-1).mean())

        # CPU float64 copy: exact-precision FD and MPS-vs-f64 gradient agreement
        nf = sc["n_f64_recipients"]
        cfg64 = copy.deepcopy(cfg)
        cfg64["model"]["dtype"] = "float64"
        m64, _ = load_model(cfg64, torch.device("cpu"))
        b64 = make_batch([p["rec_ids"] for p in rows[:nf]], ctx.pad, torch.device("cpu"))
        G64, m064, _, ld64 = probe_vjps(m64, ctx.layer, b64, Uc64)
        res["f64_leaf_logits_max_abs"] = ld64
        res["mps_vs_f64_grad_rel"] = max(_rel(G32[i], G64[i]) for i in range(nf))
        fd64 = {}
        for vname, v in (("random", v_rand[:nf]), ("delta_b", v_db[:nf])):
            an = torch.einsum("bjd,bd->bj", G64, v)
            for eps in sc["fd_eps_f64"]:
                lp = leaf_logits(m64, ctx.layer, b64, m064 + eps * v)
                lm = leaf_logits(m64, ctx.layer, b64, m064 - eps * v)
                num = (lp - lm) @ Uc64.T / (2 * eps)
                fd64[f"f64|{vname}|eps={eps}"] = max(_rel(num[i], an[i]) for i in range(nf))
        res["fd_rel_err_f64"] = fd64
        del m64

        # exact JVP vs the 8-probe estimate (technical only; never replaces the registered estimator)
        nj = sc["n_jvp_recipients"]
        jv = exact_jvp(ctx.model, ctx.layer, br, m0, db.float().to(dev)).cpu().to(F64)[:nj]
        exact = jv.norm(dim=-1) / s_cal
        est = torch.tensor([f["d_exact"] for f in feats if f["alpha"] == 1.0][:nj], dtype=F64)
        res["exact_jvp_technical"] = {"exact_norm_over_s": exact.tolist(), "probe8_D": est.tolist(),
                                      "ratio_probe_over_exact": (est / exact).tolist()}

        # descriptive old-CAL association (technical; no criterion)
        Y = np.array([f["Y"] for f in feats])
        cor = lambda k: float(np.corrcoef([f[k] for f in feats], Y)[0, 1])  # noqa: E731
        res["oldcal_descriptive_pearson_with_Y"] = {k: cor(k) for k in ("d_exact", "d_taylor", "d_shuf",
                                                                       "delta_b_norm", "b_iso", "dcurv_norm")}
        res["oldcal_probe_half_pearson"] = float(np.corrcoef([f["d_exact_half1"] for f in feats],
                                                             [f["d_exact_half2"] for f in feats])[0, 1])
        res["oldcal_Y_mean"] = float(Y.mean())
        write_csv(run.dir / "smoke_rows.csv", feats, mode="w")

        # timings / memory / projection to the registered 336 recipients x 3 alphas
        n_all = sum(cfg["design"]["expected_recipients"].values())
        res["timing"] = {"unperturbed_plus_vjp_s": t_unp, "labels_s": t_lab,
                         "per_recipient_unperturbed_plus_vjp_s": t_unp / len(rows),
                         "per_recipient_labels_3alpha_s": t_lab / len(rows),
                         "projected_all_splits_s": (t_unp + t_lab) / len(rows) * n_all,
                         "n_vjp_total_registered": n_all * cfg["probes"]["n"],
                         "n_interventions_total_registered": n_all * len(cfg["intervention"]["alphas"])}
        res["memory"] = memory_snapshot(dev)
        if dev.type == "mps":
            res["memory"]["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()

        gt = sc["gates"]
        checks = {
            "leaf_logits": res["leaf_logits_max_abs"] <= gt["leaf_logits_max_abs"],
            "fd_f32": max(fd.values()) <= gt["fd_rel_f32"],
            "fd_f64": max(fd64.values()) <= gt["fd_rel_f64"],
            "mps_vs_f64_grad": res["mps_vs_f64_grad_rel"] <= gt["mps_vs_f64_grad_rel"],
            "local_b": max(res["local_b_rel_err_unperturbed"], res["local_b_rel_err_alpha1"]) <= gt["local_b_rel"],
            "local_delta_b": res["local_delta_b_rel_err_alpha1"] <= gt["local_delta_b_rel"],
            "self_patch": res["self_patch_max_abs"] <= gt["self_patch_max_abs"],
            "other_positions": res["other_positions_max_abs"] == 0.0,
            "alpha0_delta_b": res["alpha0_delta_b_max_abs"] == 0.0,
            "same_shape_rerun": res["same_shape_rerun_G_max_abs"] <= gt["same_shape_rerun_G_max_abs"],
        }
        res["gates"] = {k: bool(v) for k, v in checks.items()}
        res["all_gates_pass"] = all(checks.values())
        run.save_json("smoke.json", res)
        if not res["all_gates_pass"]:
            raise RuntimeError(f"smoke gates failed: {[k for k, v in checks.items() if not v]}")
        return res
    return _staged("smoke", cfg, go)


def register(cfg, smoke_run, tests_log=None):
    """Write-once registration. The smoke run must be ok, with gates passed, the same source tree, manifests and
    design, and a config identical up to the protocol_version label."""
    if REGISTRATION.exists():
        raise FileExistsError(f"{REGISTRATION} exists; never overwritten")
    if cfg["protocol_version"].endswith("DRAFT"):
        raise RuntimeError("finalise the protocol_version label before registration")
    d, man = load_run(smoke_run, "smoke")
    sm = json.loads((d / "smoke.json").read_text())
    if not sm["all_gates_pass"]:
        raise RuntimeError("smoke gates did not pass")
    cur = bound_hashes(cfg)
    for k in ("source", "scp_dependency_source", "manifests", "design_sha256", "day3_config_sha256",
              "artifact_file_sha256"):
        if man["bound"][k] != cur[k]:
            raise RuntimeError(f"{k} changed since the smoke run {smoke_run}")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("protocol_version", "_path")}  # noqa: E731
    smoke_cfg = json.loads((d / "snapshot" / "followup" / "config" / "followup_v1.json").read_text())
    if strip(smoke_cfg) != strip(cfg):
        raise RuntimeError("config differs from the smoke run beyond the protocol_version label")
    if not cur["manifests"]:
        raise RuntimeError("no follow-up manifests; run `prepare` first")
    reg = {"created_utc": utcnow().isoformat(), "protocol_version": cfg["protocol_version"], "bound": cur,
           "smoke_run": smoke_run, "smoke_json_sha256": sha256_file(d / "smoke.json"),
           "tests_log": tests_log, "tests_log_sha256": sha256_file(ROOT / tests_log) if tests_log else None,
           "rule": "no fit/validation/test outcome measured before this file; any bound change invalidates it"}
    write_once(REGISTRATION, reg)
    return reg


def _measure_split(ctx, split):
    rows, _ = load_split(split)
    with ctx.run.timer(f"{split}_unperturbed_vjp"):
        unp = ctx.unperturbed(rows)
    torch.save({"rids": [p["rid"] for p in rows], "G": unp["G"]}, ctx.run.dir / f"{split}_probe_vjps.pt")
    return rows, unp


def fit(cfg):
    reg = verify_registration(cfg)

    def go(run):
        ctx = Ctx(cfg, run, labels_allowed=True)
        rows, unp = _measure_split(ctx, "fit")
        with run.timer("fit_labels"):
            labs = ctx.labels(rows, unp, "fit")
        s = scale_from_fit(labs)
        feats = join_labels(feature_rows(cfg, ctx.local, ctx.m_B, rows, unp, s, ctx.perm), labs, s)
        write_csv(run.dir / "fit_rows.csv", feats)
        y = np.array([f["Y"] for f in feats])
        models = {name: {str(lam): ridge_fit(design_matrix(feats, model_features(cfg, name)), y, lam)
                         for lam in cfg["ridge"]["lambdas"]} for name in cfg["features"]["models"]}
        h1, h2 = [f["d_exact_half1"] for f in feats], [f["d_exact_half2"] for f in feats]
        out = {"s": s, "fit_mean_Y": float(y.mean()), "n_rows": len(feats), "models": models,
               "features": {n: model_features(cfg, n) for n in cfg["features"]["models"]},
               "probe_half_reliability_descriptive": {"pearson": float(np.corrcoef(h1, h2)[0, 1]),
                                                      "mean_ratio_half1_over_half2": float(np.mean(h1) / np.mean(h2))},
               "leaf_logits_max_abs": unp["leaf_logits_max_abs"], "m0_max_abs": unp["m0_max_abs"],
               "registration_sha256": sha256_file(REGISTRATION)}
        run.save_json("fit.json", out)
        return out
    return _staged("fit", cfg, go, {"registration_sha256": sha256_file(REGISTRATION), "reg_created": reg["created_utc"]})


def validate(cfg, fit_run):
    reg = verify_registration(cfg)
    fd, fman = load_run(fit_run, "fit")
    _check_run_bound(fman, reg)
    fitj = json.loads((fd / "fit.json").read_text())

    def go(run):
        ctx = Ctx(cfg, run, labels_allowed=True)
        rows, unp = _measure_split(ctx, "validation")
        with run.timer("validation_labels"):
            labs = ctx.labels(rows, unp, "validation")
        s = fitj["s"]
        feats = join_labels(feature_rows(cfg, ctx.local, ctx.m_B, rows, unp, s, ctx.perm), labs, s)
        write_csv(run.dir / "validation_rows.csv", feats)
        y = np.array([f["Y"] for f in feats])
        mse, chosen = {}, {}
        for name in cfg["features"]["models"]:
            X = design_matrix(feats, model_features(cfg, name))
            mse[name] = {lam: weighted_mse(cfg, feats, (ridge_predict(m, X) - y) ** 2)
                         for lam, m in ((float(k), v) for k, v in fitj["models"][name].items())}
            chosen[name] = select_lambda(mse[name], cfg["ridge"]["tie_rel_tol"])
        fm = weighted_mse(cfg, feats, (max(fitj["fit_mean_Y"], 0.0) - y) ** 2)
        primary = cfg["features"]["primary"]
        beats = {n: mse[n][chosen[n]] < fm for n in primary}
        out = {"val_mse": {n: {str(k): v for k, v in d.items()} for n, d in mse.items()},
               "chosen_lambda": chosen, "fit_mean_val_mse": fm, "beats_fit_mean": beats,
               "gate_pass": any(beats.values()),
               "gate_rule": "stop before test if neither chosen primary model (A, B) beats the fit-mean predictor",
               "fit_run": fit_run}
        run.save_json("validation.json", out)
        return out
    return _staged("validate", cfg, go, {"registration_sha256": sha256_file(REGISTRATION), "fit_run": fit_run})


def freeze(cfg, fit_run, val_run):
    reg = verify_registration(cfg)
    if FROZEN.exists():
        raise FileExistsError(f"{FROZEN} exists; never overwritten")
    fd, fman = load_run(fit_run, "fit")
    vd, vman = load_run(val_run, "validate")
    _check_run_bound(fman, reg)
    _check_run_bound(vman, reg)
    if vman.get("fit_run") != fit_run:
        raise RuntimeError("validation run used a different fit run")
    fitj = json.loads((fd / "fit.json").read_text())
    val = json.loads((vd / "validation.json").read_text())
    selected = {n: {"lambda": val["chosen_lambda"][n], **fitj["models"][n][str(val["chosen_lambda"][n])]}
                for n in cfg["features"]["models"]}
    fr = {"created_utc": utcnow().isoformat(), "registration_sha256": sha256_file(REGISTRATION),
          "source_tree_sha256": reg["bound"]["source"]["tree_sha256"], "s": fitj["s"],
          "fit_mean_Y": fitj["fit_mean_Y"], "features": fitj["features"], "selected": selected,
          "validation_gate_pass": val["gate_pass"], "proceed_to_test": bool(val["gate_pass"]),
          "fit_run": fit_run, "val_run": val_run, "fit_outputs": fman["outputs"], "val_outputs": vman["outputs"]}
    write_once(FROZEN, fr)
    return fr


def predict_test(cfg):
    reg = verify_registration(cfg)
    fr = load_frozen(cfg, reg)
    if not fr["proceed_to_test"]:
        raise RuntimeError("validation gate failed: development NO-GO, no test evaluation")
    if TEST_LOCK.exists():
        raise FileExistsError(f"{TEST_LOCK} exists; test predictions are made once")

    def go(run):
        ctx = Ctx(cfg, run, labels_allowed=False)
        rows, unp = _measure_split(ctx, "test")
        feats = feature_rows(cfg, ctx.local, ctx.m_B, rows, unp, fr["s"], ctx.perm)
        write_csv(run.dir / "test_features.csv", feats)
        preds = []
        Xs = {n: design_matrix(feats, fr["features"][n]) for n in fr["selected"]}
        P = {n: ridge_predict(fr["selected"][n], Xs[n]) for n in fr["selected"]}
        for i, f in enumerate(feats):
            preds.append({"rid": f["rid"], "alpha": f["alpha"], "template": f["template"],
                          **{f"pred_{n}": float(P[n][i]) for n in P}, "pred_fit_mean": max(fr["fit_mean_Y"], 0.0)})
        write_csv(run.dir / "test_predictions.csv", preds)
        lock = {"created_utc": utcnow().isoformat(), "run_id": run.id,
                "test_features_sha256": sha256_file(run.dir / "test_features.csv"),
                "test_predictions_sha256": sha256_file(run.dir / "test_predictions.csv"),
                "test_probe_vjps_sha256": sha256_file(run.dir / "test_probe_vjps.pt"),
                "frozen_sha256": sha256_file(FROZEN), "registration_sha256": sha256_file(REGISTRATION),
                "rule": "written before any perturbed test forward; evaluate-test verifies it first"}
        write_once(TEST_LOCK, lock)
        return lock
    return _staged("predict_test", cfg, go, {"registration_sha256": sha256_file(REGISTRATION)})


def _read_float_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def evaluate_test(cfg):
    reg = verify_registration(cfg)
    fr = load_frozen(cfg, reg)
    if not TEST_LOCK.exists():
        raise RuntimeError("no test prediction lock; run `predict-test` first")
    if TEST_EVALUATED.exists():
        raise FileExistsError("test labels were already evaluated once")
    lock = json.loads(TEST_LOCK.read_text())
    pd_, pman = load_run(lock["run_id"], "predict_test")
    _check_run_bound(pman, reg)
    for k, name in (("test_features_sha256", "test_features.csv"), ("test_predictions_sha256", "test_predictions.csv"),
                    ("test_probe_vjps_sha256", "test_probe_vjps.pt")):
        if sha256_file(pd_ / name) != lock[k]:
            raise RuntimeError(f"{name} changed after the lock")
    if lock["frozen_sha256"] != sha256_file(FROZEN):
        raise RuntimeError("freeze file changed after the lock")

    def go(run):
        feats = _read_float_csv(pd_ / "test_features.csv")
        preds = {(p["rid"], p["alpha"]): p for p in _read_float_csv(pd_ / "test_predictions.csv")}
        ctx = Ctx(cfg, run, labels_allowed=True)
        rows, _ = load_split("test")
        unp = ctx.unperturbed(rows, with_vjp=False)
        # reproduction of the locked, label-free features from fresh unperturbed forwards
        kl_now = {p["rid"]: float(unp["kl"][i]) for i, p in enumerate(rows)}
        repro = max(abs(kl_now[f["rid"]] - f["kl_unpert"]) / max(abs(f["kl_unpert"]), 1e-12) for f in feats)
        run.manifest["feature_reproduction_kl_max_rel"] = repro
        if repro > 1e-5:
            raise RuntimeError(f"unperturbed test forwards do not reproduce the locked features ({repro:.2e})")
        with run.timer("test_labels"):
            labs = ctx.labels(rows, unp, "test")
        write_csv(run.dir / "test_labels.csv", labs)
        s = fr["s"]
        lab = {(r["rid"], r["alpha"]): r["label_raw"] / s for r in labs}
        rowsY = [{**f, "Y": lab[(f["rid"], f["alpha"])]} for f in feats]
        y = np.array([r["Y"] for r in rowsY])
        err2 = {n: (np.array([preds[(r["rid"], r["alpha"])][f"pred_{n}"] for r in rowsY]) - y) ** 2
                for n in list(fr["selected"]) + ["fit_mean"]}
        prim = gain(cfg, rowsY, err2["A"], err2["B"])
        prim["decision"] = decision(cfg, prim)
        sec = gain(cfg, rowsY, err2["A_in"], err2["B"])
        desc = {"per_alpha": {}, "per_frame": {}}
        for key, dim in (("per_alpha", "alpha"), ("per_frame", "frame")):
            for lv in sorted({r[dim] for r in rowsY}):
                m = np.array([r[dim] == lv for r in rowsY])
                mA, mB = err2["A"][m].mean(), err2["B"][m].mean()
                desc[key][str(lv)] = {"mse_A": float(mA), "mse_B": float(mB),
                                      "G_point": float(1 - mA / mB) if mB > cfg["evaluation"]["mse_floor"] else None}
        out = {"primary": {"comparison": "A = B + D_exact vs B", **prim},
               "secondary_descriptive": {"comparison": "A_in = B + D_Taylor vs B", **sec},
               "weighted_mse": {n: weighted_mse(cfg, rowsY, e) for n, e in err2.items()},
               "descriptive": desc, "Y_mean": float(y.mean()), "s": s, "n_rows": len(rowsY),
               "lock": lock, "frozen_sha256": sha256_file(FROZEN)}
        run.save_json("results.json", out)
        write_once(TEST_EVALUATED, {"created_utc": utcnow().isoformat(), "run_id": run.id,
                                    "results_sha256": sha256_file(run.dir / "results.json")})
        return out
    return _staged("evaluate_test", cfg, go, {"registration_sha256": sha256_file(REGISTRATION)})


def report(cfg, eval_run):
    """Render an evaluate-test run's results.json into REPORT.md inside a new report run (the evaluated run's
    outputs are never touched)."""
    d, _ = load_run(eval_run, "evaluate_test")
    r = json.loads((d / "results.json").read_text())

    def go(run):
        p, s2 = r["primary"], r["secondary_descriptive"]
        fmt = lambda x: "n/a" if x is None else f"{x:.6g}"  # noqa: E731
        lines = [f"# Follow-up test report ({eval_run})", "",
                 f"Primary (A = B + D_exact vs B): G = {fmt(p['G'])}; one-sided {p['one_sided_conf']:.0%} lower "
                 f"{fmt(p['G_lower_one_sided'])}, upper {fmt(p['G_upper_one_sided'])}; MSE_A {fmt(p['mse_A'])}, "
                 f"MSE_B {fmt(p['mse_B'])}, MSE_B - MSE_A {fmt(p['mse_B_minus_A'])} "
                 f"(two-sided 95% {p['mse_B_minus_A_ci95_two_sided']}). Decision: **{p['decision']}**.", "",
                 f"Secondary, descriptive (A_in = B + D_Taylor vs B): G = {fmt(s2['G'])}; lower "
                 f"{fmt(s2['G_lower_one_sided'])}, upper {fmt(s2['G_upper_one_sided'])}.", "",
                 "Limits: fixed 8 probes (probe uncertainty is not in the template bootstrap); one model, layer, rank "
                 "and task; alpha 1.5 is an extrapolation; nominal, approximate, conditional intervals; not "
                 "mechanism evidence.", ""]
        (run.dir / "REPORT.md").write_text("\n".join(lines))
        return run.dir / "REPORT.md"
    return _staged("report", cfg, go, {"eval_run": eval_run, "results_sha256": sha256_file(d / "results.json")})
