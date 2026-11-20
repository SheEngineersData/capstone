"""Full-network federated simulation (Keras): FedAvg and FedProx, Dirichlet label-skew clients,
client-side differential privacy and update compression.

Every trainable layer of the MobileNetV2 is trained on the simulated clients and averaged by the
server. BatchNorm layers stay in inference mode with their v0 statistics and are not exchanged.
The update scope can be limited to the last blocks ('partial') or the classifier ('head') for the
cost comparison. Runs checkpoint every round and resume after a crash or reboot.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from . import config as C
from . import model as M

keras, tf = M.keras, M.tf
PARTIAL_SHARE = 0.10  # partial scope: smallest set of last layers whose update is about a tenth of the full update


# -------------------------------------------------------------- partitioning
def dirichlet_partition(labels, idx, n_clients, alpha, rng, min_size=10, max_tries=2000):
    """Split `idx` across clients. For each class a Dirichlet(alpha) draw sets each client's share.
    Small alpha gives clients dominated by a few classes. Redrawn until every client has >= min_size."""
    labels = np.asarray(labels)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(labels[idx]):
            ci = rng.permutation(idx[labels[idx] == c])
            share = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(share) * len(ci)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(ci, cuts)):
                parts[k].extend(chunk.tolist())
        if min(map(len, parts)) >= min_size:
            return [np.array(p) for p in parts]
    raise RuntimeError("could not satisfy min_size; raise alpha or lower n_clients")


def label_skew(parts, y, n_classes):
    """Mean Jensen-Shannon divergence between each client's label mix and the global mix (0 = IID)."""
    g = np.bincount(np.concatenate([y[p] for p in parts]), minlength=n_classes).astype(float); g /= g.sum()
    out = []
    for p in parts:
        q = np.bincount(y[p], minlength=n_classes).astype(float); q /= q.sum()
        m = (q + g) / 2
        kl = lambda a, b: np.sum(a[a > 0] * np.log(a[a > 0] / np.maximum(b[a > 0], 1e-12)))
        out.append(0.5 * kl(q, m) + 0.5 * kl(g, m))
    return float(np.mean(out))


# ------------------------------------------------- update compression and privacy
def compress(delta, method, rate=0.1):
    """Returns (reconstructed delta, bytes on the wire). Delta is a flat float32 vector."""
    n = delta.size
    if method in (None, "none"):
        return delta, n * 4
    if method == "int8":
        s = np.abs(delta).max() / 127 or 1.0
        return np.round(delta / s).clip(-127, 127) * s, n + 4
    if method == "topk":
        k = max(1, int(rate * n))
        keep = np.argpartition(np.abs(delta), -k)[-k:]
        out = np.zeros_like(delta); out[keep] = delta[keep]
        return out, k * 8  # 4 byte value + 4 byte index
    raise ValueError(method)


def privatize(delta, clip, noise_mult, rng):
    """Client-side DP: clip the update's L2 norm to `clip`, add Gaussian noise with std noise_mult*clip."""
    nrm = np.linalg.norm(delta)
    delta = delta * min(1.0, clip / (nrm + 1e-12))
    return delta + rng.normal(0, noise_mult * clip, delta.shape).astype(delta.dtype)


# ---------------------------------------------------------------- model scopes
def trainable_subset(model, scope):
    """Set which layers train for `scope` in {'full', 'partial', 'head'}. BatchNorm is never trained."""
    M.set_trainable(model, "all" if scope == "full" else "head")
    if scope != "partial":
        return
    base = M.get_base(model)
    layers = [l for l in base.layers if l.weights and not isinstance(l, keras.layers.BatchNormalization)]
    size = lambda l: sum(int(np.prod(w.shape)) for w in l.weights)
    head_n = sum(int(np.prod(w.shape)) for w in model.get_layer("head").weights)
    total = sum(size(l) for l in layers) + head_n
    base.trainable = True  # sets every sublayer; the per-layer flags below then narrow it down
    for l in base.layers:
        l.trainable = False
    got = head_n
    for l in reversed(layers):
        if got >= PARTIAL_SHARE * total:
            break
        l.trainable = True
        got += size(l)


def get_vec(model):
    return np.concatenate([v.numpy().ravel() for v in model.trainable_variables]).astype(np.float32)


def set_vec(model, vec):
    i = 0
    for v in model.trainable_variables:
        n = int(np.prod(v.shape))
        v.assign(vec[i:i + n].reshape(v.shape))
        i += n


def start_model(num_classes, init, seed, scope):
    """init B: Stage 1 pool model v0. init A: ImageNet backbone with an empty (zero) head, no potato knowledge."""
    model = M.build_model(num_classes, pretrained=True)
    if init == "B":
        model.load_weights(str(C.MODELS / "v0" / f"seed{seed}" / "best.weights.h5"))
    else:
        head = model.get_layer("head")
        head.set_weights([np.zeros_like(w) for w in head.get_weights()])
    trainable_subset(model, scope)
    return model


# ---------------------------------------------------------------- local training
class Trainer:
    """Local SGD for one client. One optimizer per run, its state reset for every client and round."""

    def __init__(self, model, lr):
        self.model = model
        self.vars = model.trainable_variables
        self.opt = keras.optimizers.Adam(lr)
        self.opt.build(self.vars)
        self.glob = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.vars]
        self.mu = tf.Variable(0.0, trainable=False)
        self._step = tf.function(self._step_impl)

    def _step_impl(self, x, y, sw):
        with tf.GradientTape() as tape:
            p = self.model(x, training=True)  # dropout on, BatchNorm in inference mode (not trainable)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y, p)
            loss = tf.reduce_mean(ce * sw)
            prox = tf.add_n([tf.reduce_sum(tf.square(v - g)) for v, g in zip(self.vars, self.glob)])
            loss = loss + 0.5 * self.mu * prox
        self.opt.apply_gradients(zip(tape.gradient(loss, self.vars), self.vars))
        return loss

    def reset(self):
        for v in self.opt.variables:
            v.assign(tf.zeros_like(v))

    def train(self, X, y, idx, epochs, batch, seed, mu, weighted=True):
        self.reset()
        self.mu.assign(mu)
        for v, g in zip(self.vars, self.glob):
            g.assign(v)
        cw = M.class_weights(y, idx) if weighted else None
        ds = M.make_ds(X, y, idx, batch, train=True, seed=seed)
        for _ in range(epochs):
            for xb, yb in ds:
                if len(yb) < 2:
                    continue
                sw = tf.constant([cw[int(c)] for c in yb.numpy()], tf.float32) if cw else tf.ones_like(tf.cast(yb, tf.float32))
                self._step(xb, yb, sw)


# ------------------------------------------------------------------------ server
def run_full(X, y, parts, val_idx, test_idx, n_classes, init, seed, scope="full", rounds=30, local_epochs=1, lr=1e-4, batch=32,
             mu=0.0, frac=1.0, drop_p=0.0, comp=None, comp_rate=0.1, dp=None, weighted=True, eval_every=5, week_frac=1.0,
             save_to=None, ckpt=None):
    """One federated run on the whole network (or a scope of it). Returns a dict like fl_head.run_fl."""
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = start_model(n_classes, init, seed, scope)
    tr = Trainer(model, lr)
    w = get_vec(model)
    n_params = w.size
    K = len(parts)
    hist, up_total, down_total, r0 = [], 0, 0, 1
    t0 = time.time()
    if ckpt and Path(ckpt).exists():  # resume after a crash or reboot
        c = pickle.load(open(ckpt, "rb"))
        w, hist, up_total, down_total, r0 = c["w"], c["hist"], c["up"], c["down"], c["next"]
        rng.bit_generator.state = c["rng"]
        t0 -= c["secs"]
        print(f"  resumed at round {r0}", flush=True)
    for r in range(r0, rounds + 1):
        chosen = rng.choice(K, max(1, int(round(frac * K))), replace=False)
        survivors = [k for k in chosen if rng.random() >= drop_p]
        deltas, ns, up = [], [], 0
        for k in survivors:
            idx = parts[k]
            if week_frac < 1.0:
                idx = rng.choice(idx, max(10, int(week_frac * len(idx))), replace=False)
            set_vec(model, w)
            tr.train(X, y, idx, local_epochs, batch, seed + 1000 * r + int(k), mu, weighted)
            d = get_vec(model) - w
            if dp:
                local = dp.get("mode", "local") == "local"
                d = privatize(d, dp["clip"], dp["noise"] if local else 0.0, rng)
            d, nbytes = compress(d, comp, comp_rate)
            deltas.append(d)
            ns.append(len(parts[k]))
            up += nbytes
        if deltas:
            wt = np.ones(len(ns)) if dp else np.array(ns, float)
            agg = np.tensordot(wt / wt.sum(), np.stack(deltas), axes=1).astype(np.float32)
            if dp and dp.get("mode") == "central":
                agg = agg + rng.normal(0, dp["noise"] * dp["clip"] / len(deltas), agg.shape).astype(np.float32)
            w = w + agg
        up_total += up
        down_total += len(chosen) * n_params * 4
        if r % eval_every == 0 or r == rounds:
            set_vec(model, w)
            va = M.evaluate(model, X, y, val_idx)
            hist.append({"round": r, "val_acc": va["acc"], "val_macro_f1": va["macro_f1"], "participants": len(survivors),
                         "bytes_up_total": up_total, "bytes_down_total": down_total, "secs": time.time() - t0})
            print(f"  round {r:2d} val_f1 {va['macro_f1']:.3f} acc {va['acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if ckpt:
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({"w": w, "hist": hist, "up": up_total, "down": down_total, "next": r + 1, "rng": rng.bit_generator.state, "secs": time.time() - t0},
                        open(str(ckpt) + ".tmp", "wb"))
            Path(str(ckpt) + ".tmp").replace(ckpt)
    set_vec(model, w)
    te = M.evaluate(model, X, y, test_idx)
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(save_to))
    if ckpt and Path(ckpt).exists():
        Path(ckpt).unlink()
    return {"history": hist, "test_acc": te["acc"], "test_macro_f1": te["macro_f1"], "n_params": n_params, "preds": te["preds"]}


