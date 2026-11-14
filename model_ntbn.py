"""
model_ntbn.py: NTBN Model and Rollout Implementation in JAX (AuditFix Edition).
Fixes action distortion:
1. Pure NTBN policy: delta_target = clamp(delta_prev, bl, bu), action = delta_target - delta_prev.
2. action_clipping:
   - "none" (default): Pure NTBN policy without any softclip distortion.
   - "hard": Physical trading bounds via hard clipping jnp.clip(a, lb, ub), zero distortion within bounds.
   - "soft": Buehler's DHSoftClip approximation, kept only for diagnostic/ablation comparison.
3. init_delta:
   - Default is False in pure NTBN (hedger starts with delta_0 = 0, first trade brings position into [bl, bu]).
   - If True, idelta is maintained for compatibility/ablation, but documented to displace delta_1 outside band.
4. Cost calculation:
   - market["cost"] is already S_t * c_s (dollar cost per unit traded).
   - cost = cost + sum(|a| * m["cost"]) is identically sum(|a_t| * S_t * c_s).
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import jax
import jax.numpy as jnp

from .ntbn_core import ntbn_clamp, ntbn_bounds_center_width, ntbn_action

STATE = "recurrent_state"
UTILITIES = ("mean", "exp", "exp2", "vicky", "cvar", "quad")
ACTIVATIONS = dict(
    relu=jax.nn.relu,
    softplus=jax.nn.softplus,
    tanh=jnp.tanh,
    sigmoid=jax.nn.sigmoid,
    linear=lambda x: x
)


@dataclass(frozen=True)
class AgentConfigNTBN:
    """
    Configuration for NTBN Agent.
    Note: 'delta' and 'action' are excluded from neural network inputs (action-independence).
    """
    features: tuple = ("price", "time_left")
    width: int = 20
    depth: int = 3
    activation: str = "relu"
    band_mode: str = "center_width"  # "center_width" (w >= 0) or "direct"
    classic: int = 0
    aggregate: int = 0
    past_repr: int = 0
    event: int = 0
    bound_aggr_states: bool = False
    init_delta: bool = False         # Pure NTBN default is False (starts with delta_0 = 0)
    action_clipping: str = "none"    # "none" (pure NTBN), "hard" (jnp.clip), "soft" (softclip)

    @property
    def n_states(self) -> int:
        return self.classic + self.aggregate + self.past_repr + self.event


@dataclass(frozen=True)
class GymConfigNTBN:
    agent: AgentConfigNTBN = field(default_factory=AgentConfigNTBN)
    utility: str = "exp2"
    lmbda: float = 1.0


def ntbn_input_features(cfg: AgentConfigNTBN) -> list:
    """
    Returns sorted list of exogenous features.
    Filters out 'delta' and 'action' to eliminate action-dependence.
    """
    exogenous = {f for f in cfg.features if f not in ("delta", "action")}
    if cfg.n_states:
        exogenous.add(STATE)
    return sorted(exogenous)


def _glorot(key, shape):
    fan_in, fan_out = (shape[0], shape[0]) if len(shape) == 1 else shape
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, jnp.float32, -lim, lim)


def _unit(x):
    return jnp.where(x > 0.5, 1.0, 0.0)


def _split(x, sizes):
    out, start = [], 0
    for s in sizes:
        out.append(x[:, start:start + s])
        start += s
    return out


def init_params_ntbn(key, cfg: GymConfigNTBN, feature_dims: dict, n_inst: int) -> dict:
    """
    Initializes parameters for NTBN.
    Network outputs 2 * n_inst (for lower and upper bands / center and width).
    """
    a = cfg.agent
    dims = dict(feature_dims, **({STATE: a.n_states} if a.n_states else {}))
    in_dim = sum(dims[f] for f in ntbn_input_features(a))
    
    # 2 * n_inst for bands, plus any recurrent state outputs
    out_dim = 2 * n_inst + a.n_states + a.aggregate + a.past_repr + a.event
    sizes = [in_dim] + [a.width] * a.depth + [out_dim]
    
    keys = jax.random.split(key, len(sizes) + 1)
    dense_layers = [
        dict(W=_glorot(k, (i, o)), b=jnp.zeros((o,), jnp.float32))
        for k, i, o in zip(keys[:-2], sizes[:-1], sizes[1:])
    ]
    
    params = dict(
        dense=dense_layers,
        y=jnp.zeros((1,), jnp.float32),
        y0=jnp.zeros((1,), jnp.float32)
    )
    if a.n_states:
        params["state0"] = _glorot(keys[-2], (a.n_states,))
    if a.init_delta:
        params["idelta"] = _glorot(keys[-1], (n_inst,))
    return params


def agent_step_ntbn(p: dict, cfg: AgentConfigNTBN, act, feats: dict, state: jnp.ndarray, n_inst: int) -> tuple:
    """
    Single-step forward pass for NTBN.
    Inputs: exogenous market features and recurrent state (no delta/action inputs).
    Outputs: ((bl, bu), new_state).
    """
    feats = dict(feats)
    if cfg.n_states:
        classic, aggr, past, event = _split(state, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event))
        classic, event = jnp.tanh(classic), _unit(event)
        aggr = jnp.tanh(aggr) if cfg.bound_aggr_states else aggr
        feats[STATE] = jnp.concatenate([classic, aggr, past, event], axis=1)
        
    z = jnp.concatenate([feats[f] for f in ntbn_input_features(cfg)], axis=1)
    for layer in p["dense"][:-1]:
        z = act(z @ layer["W"] + layer["b"])
    out = z @ p["dense"][-1]["W"] + p["dense"][-1]["b"]
    
    raw_bounds = out[:, :2 * n_inst]
    if cfg.band_mode == "center_width":
        bl, bu = ntbn_bounds_center_width(raw_bounds, n_inst)
    else:  # direct
        bl, bu = raw_bounds[:, :n_inst], raw_bounds[:, n_inst:2 * n_inst]

    if not cfg.n_states:
        return (bl, bu), state

    rec = out[:, 2 * n_inst:]
    c_new, a_cand, p_cand, e_cand, a_upd, p_upd, e_upd = _split(
        rec, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event, cfg.aggregate, cfg.past_repr, cfg.event)
    )
    a_cand = jnp.tanh(a_cand) if cfg.bound_aggr_states else a_cand
    a_upd, p_upd, e_upd = jax.nn.sigmoid(a_upd), _unit(p_upd), _unit(e_upd)
    new_state = jnp.concatenate([
        jnp.tanh(c_new),
        (1.0 - a_upd) * aggr + a_upd * a_cand,
        (1.0 - p_upd) * past + p_upd * p_cand,
        (1.0 - e_upd) * event + e_upd * _unit(e_cand),
    ], axis=1)
    return (bl, bu), new_state


def softclip(a, lb, ub):
    """
    Original Buehler DHSoftClip smooth saturation function.
    Warning: Distorts a even when a=0 (e.g. for [-5, 5], softclip(0) = -0.3774).
    Kept exclusively for ablation and diagnostic comparison.
    """
    sp = jax.nn.softplus
    a = jnp.minimum(a, ub * 10.0)
    a = jnp.maximum(a, lb * 10.0)
    d = ub - lb
    r = (a - lb) / d
    r = 1.0 - sp(1.0 - sp(r)) / sp(1.0)
    return jnp.where(d > 0.0, r * d + lb, 0.0)


def _neg(g):
    return jnp.where(g < 0.0, g, 0.0)


def utility(name: str, lmbda: float, X, y):
    if lmbda < 1e-12:
        name, lmbda = "mean", 1.0
    gains = X + y
    if name == "mean":
        return gains
    if name == "cvar":
        return (1.0 + lmbda) * _neg(gains) - y
    if name == "quad":
        x0 = 1.0 / lmbda
        xx = _neg(gains - x0)
        return -0.5 * lmbda * xx ** 2 + 0.5 * lmbda * x0 ** 2 - y
    if name == "exp":
        inf = jax.lax.stop_gradient(jnp.min(X))
        return (1.0 - jnp.exp(-lmbda * (gains - inf))) / lmbda - y + inf
    if name == "exp2":
        g1 = jnp.where(gains >= 0.0, gains, 0.0)
        g2 = jnp.where(gains <= 0.0, gains, 0.0)
        u1 = (1.0 - jnp.exp(-lmbda * g1)) / lmbda - y
        u2 = g2 - 0.5 * lmbda * g2 * g2 - y
        return jnp.where(gains > 0.0, u1, u2)
    if name == "vicky":
        return (1.0 + lmbda * gains - jnp.sqrt(1.0 + (lmbda * gains) ** 2)) / lmbda - y
    raise ValueError(f"Unknown utility '{name}'. Use one of {UTILITIES}")


def rollout_ntbn(p: dict, cfg: GymConfigNTBN, batch: dict) -> dict:
    """
    Rollout trajectories under the NTBN policy.
    At each step:
      (bl, bu) = NN(I_t)
      delta_target = clamp(delta_prev, bl, bu)
      a_target = delta_target - delta_prev
      
    Action determination:
      - action_clipping == 'none': a = a_target (pure NTBN)
      - action_clipping == 'hard': a = jnp.clip(a_target, lb, ub)
      - action_clipping == 'soft': a = softclip(a_target, lb, ub) (Buehler ablation)
      
    If cfg.agent.init_delta is True, idelta * first is added to action at step 0.
    In pure NTBN, init_delta is False, ensuring the first trade is purely dictated by [bl(0), bu(0)].
    """
    a_cfg, act = cfg.agent, ACTIVATIONS[cfg.agent.activation]
    m = batch["market"]
    payoff, hedges = m["payoff"], m["hedges"]
    n, _, n_inst = hedges.shape
    zeros_i = jnp.zeros((n, n_inst), jnp.float32)
    state0 = jnp.broadcast_to(p["state0"], (n, a_cfg.n_states)) if a_cfg.n_states else jnp.zeros((n, 0), jnp.float32)
    idelta = p["idelta"][None, :] if a_cfg.init_delta else zeros_i
    
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    xs = (tm(hedges), tm(m["cost"]), tm(m["lbnd_a"]), tm(m["ubnd_a"]), {k: tm(v) for k, v in batch["features"].items()})

    def body(carry, x):
        delta, action, cost, pnl, state, first = carry
        h, c, lb, ub, f = x
        
        # 1. NTBN predicts bands [bl, bu] from exogenous features f
        (bl, bu), state = agent_step_ntbn(p, a_cfg, act, f, state, n_inst)
        
        # 2. Clamping policy: delta_target = clamp(delta, bl, bu)
        delta_target, a_target = ntbn_action(delta, bl, bu)
        
        # 3. Add initial delta bias if configured
        if a_cfg.init_delta:
            a_target = a_target + idelta * first
            
        # 4. Action determination based on action_clipping mode
        if a_cfg.action_clipping == "none":
            a = a_target
        elif a_cfg.action_clipping == "hard":
            a = jnp.clip(a_target, lb, ub)
        elif a_cfg.action_clipping == "soft":
            a = softclip(a_target, lb, ub)
        else:
            raise ValueError(f"Unknown action_clipping: {a_cfg.action_clipping}")
            
        # 5. Position update
        delta = delta + a
        
        # 6. PnL and Cost accrual:
        # Note: c is market["cost"], which already contains S_t * cost_s (dollar cost per unit traded)
        cost = cost + jnp.sum(jnp.abs(a) * c, axis=1)
        pnl = pnl + jnp.sum(a * h, axis=1)
        
        return (delta, a, cost, pnl, state, 0.0), (jax.lax.stop_gradient(a), jax.lax.stop_gradient(bl), jax.lax.stop_gradient(bu))

    z = jnp.zeros_like(payoff)
    (_, _, cost, pnl, _, _), (actions, bls, bus) = jax.lax.scan(
        body, (zeros_i, zeros_i, z, z, state0, 1.0), xs
    )
    
    y = p["y"][0] if cfg.utility != "mean" else 0.0
    y0 = p["y0"][0] if cfg.utility != "mean" else 0.0
    u = utility(cfg.utility, cfg.lmbda, payoff + pnl - cost, y)
    u0 = utility(cfg.utility, cfg.lmbda, payoff, y0)
    sg = jax.lax.stop_gradient
    
    return dict(
        loss=-u - u0,
        utility=sg(u),
        utility0=sg(u0),
        gains=sg(payoff + pnl - cost),
        payoff=payoff,
        pnl=sg(pnl),
        cost=sg(cost),
        actions=jnp.swapaxes(actions, 0, 1),
        b_lower=jnp.swapaxes(bls, 0, 1),
        b_upper=jnp.swapaxes(bus, 0, 1)
    )


def loss_ntbn(p: dict, cfg: GymConfigNTBN, batch: dict):
    return jnp.mean(rollout_ntbn(p, cfg, batch)["loss"])


def as_batch(world) -> dict:
    return dict(
        market={k: jnp.asarray(v) for k, v in world.market.items()},
        features={k: jnp.asarray(v) for k, v in world.features.items()}
    )
