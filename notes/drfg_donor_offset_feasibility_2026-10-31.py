import itertools, json
from collections import Counter
n, K = 48, 8

def mult(S):
    c = Counter((a - b) % n for a in S for b in S if a != b)
    return c

def search(max_m, want_doubles=None):
    """Lexicographically first increasing S with S[0]=0 and every nonzero ordered difference multiplicity <= max_m."""
    nodes = 0
    def rec(S):
        nonlocal nodes
        nodes += 1
        if len(S) == K:
            c = mult(S)
            if want_doubles is None or sum(v == 2 for v in c.values()) == want_doubles:
                return S
            return None
        for x in range(S[-1] + 1, n):
            T = S + [x]
            if max(mult(T).values()) <= max_m:
                r = rec(T)
                if r:
                    return r
        return None
    return rec([0]), nodes

def checks(S):
    c = mult(S)
    groups = range(n)
    sets = [frozenset((g + s) % n for s in S) for g in groups]
    ov = Counter(len(sets[g] & sets[h]) for g in groups for h in groups if g < h)
    use = Counter(x for st in sets for x in st)
    ctx = Counter(((g + S[k]) % n, k) for g in groups for k in range(K))  # (noun, frame slot)
    same_tmpl = [len(sets[2 * t] & sets[2 * t + 1]) for t in range(24)]
    same_frame = Counter(len(sets[g] & sets[h]) for f in range(4) for g in range(12 * f, 12 * f + 12)
                         for h in range(12 * f, 12 * f + 12) if g < h)
    return {"offsets": S, "distinct": len(set(S)) == K, "max_ordered_diff_mult": max(c.values()),
            "n_diffs_mult2": sum(v == 2 for v in c.values()), "n_diffs_mult1": sum(v == 1 for v in c.values()),
            "n_diffs_mult0": n - 1 - len(c), "mult_of_24": c.get(24, 0),
            "each_noun_used": sorted(set(use.values())), "contexts_unique": max(ctx.values()) == 1 and len(ctx) == 384,
            "distinct_noun_sets": len(set(sets)),
            "pairwise_overlap_distribution_1128_pairs": dict(sorted(ov.items())),
            "same_template_pair_overlaps": dict(Counter(same_tmpl)),
            "same_frame_pair_overlaps": dict(sorted(same_frame.items()))}

old = [6 * k for k in range(K)]
print("OLD", json.dumps(checks(old)))
S1, nodes1 = search(1)
print("lambda<=1 feasible:", S1, "nodes", nodes1, "| pigeonhole: 56 ordered differences > 47 nonzero residues")
S2, nodes2 = search(2)
print("FIRST max<=2", json.dumps(checks(S2)), "nodes", nodes2)
S3, nodes3 = search(2, want_doubles=9)
print("FIRST max<=2 with minimal 9 doubled residues", json.dumps(checks(S3)), "nodes", nodes3)
