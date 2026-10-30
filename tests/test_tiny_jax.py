"""flint_jax on the tiny MLX-built checkpoint (2 virtual CPU devices) must match MLX's logits, and LoRA must train."""
import os, sys, json, glob
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import numpy as np, jax, jax.numpy as jnp
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_root, "model"), os.path.join(_root, "train")]
import flint_jax as fj
d = sys.argv[1]
cfg = json.load(open(f"{d}/config.json"))
m = fj.Flint(cfg, fj.load(cfg, fj.iter_tensors(sorted(glob.glob(f"{d}/*.safetensors"))), jax.devices()))
ids = jnp.asarray(np.load(f"{d}/ids.npy")); ref = np.load(f"{d}/logits.npy")
got = np.asarray(m.logits(ids))
err = np.abs(got - ref).max() / np.abs(ref).max()
print("rel max err", err, "top1", (got.argmax(-1) == ref.argmax(-1)).mean()); assert err < 3e-2
lo = m.init_lora(4, 8)
st = (jax.tree.map(jnp.zeros_like, lo), jax.tree.map(jnp.zeros_like, lo))
losses = []
for s in range(1, 6):
    l, g = jax.value_and_grad(lambda lo: m.loss(lo, ids, chunk=6))(lo)
    lo, st = fj.adamw(lo, g, st, 1e-2, step=s); losses.append(float(l))
print("losses", [round(x, 3) for x in losses]); assert losses[-1] < losses[0]
