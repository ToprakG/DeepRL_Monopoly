import numpy as np

import engine_shim  # noqa: F401
import feature_lib as fl
from act_lib import _STATIC, SEC_NAMES, action_features
from v2_lib import _decode_trade
from feature_lib import (_COLOR_OF, _GROUP_SQUARES, _HOUSE_PRICE, GROUP_ORDER,
                         GROUP_STEP_EFFICIENCY)
from engine.constants import PROPERTY_IDS

N_ACT_F = 21
N_STATE_F = 16
N_CAND_F = 34
FEAT_DIM = 71
FEATURE_NAMES = ['act_0', 'act_1', 'act_2', 'act_3', 'act_4', 'act_5', 'act_6', 'act_7', 'act_8', 'act_9', 'act_10', 'act_11', 'act_12', 'act_13', 'act_14', 'act_15', 'act_16', 'act_17', 'act_18', 'act_19', 'act_20', 'cash', 'round_frac', 'net_worth', 'n_owned', 'own_monop', 'opp_monop_max', 'cash_reserve', 'war_chest_gap', 'liq_margin', 'unowned_frac', 'phase_open', 'phase_mid', 'rent_exposure', 'open_debt', 'build_shortfall', 'chest_active', 'val_ratio_self', 'val_abs_self', 'completes_self', 'completes_opp', 'sq_income', 'blk_income', 'is_stranded', 'price_over_cash', 'val_ratio_opp_max', 'trade_cash_lvl', 'trade_cash_over_cash', 'exch_val_in', 'exch_val_edge', 'exch_same_group', 'n_legal', 'sec_frac', 'cp_cash', 'cp_completes_main', 'cp_val_main', 'cp_val_req', 'exch_val_req_self', 'cp_index', 'mv_over_cash', 'inc_per_dollar', 'is_dev_group', 'redeem_cost_over_cash', 'redeem_margin', 'exch_cp_edge', 'raised_abs', 'build_eff', 'improve_lump_ok', 'redeem_key', 'redeem_afford', 'redeem_frozen']


def _mortgage_v(prop):
    return float(getattr(prop, 'mortgage_value', getattr(prop, 'price', 0) // 2))


def state_block(env, pid, ctx, chest_target):
    p = env.players[pid]
    cash = float(getattr(p, "money", getattr(p, "cash", 0.0)))
    liq = cash
    n_owned = own_mon = 0
    for sq, prop in env.properties.items():
        if prop.owner == pid:
            n_owned += 1
            hp = float(getattr(prop, "house_price", getattr(prop, "house_cost", 50)))
            liq += prop.houses * hp / 2.0
            if not prop.mortgaged:
                liq += _mortgage_v(prop)
            if getattr(prop, "is_monopoly", False):
                own_mon = own_mon  # counted below via groups; keep simple flag
    own_mon = sum(1 for sq, pr in env.properties.items()
                  if pr.owner == pid and getattr(pr, "is_monopoly", False))
    opp_mon = [0, 0, 0, 0]
    for sq, pr in env.properties.items():
        if pr.owner is not None and pr.owner != pid and getattr(pr, "is_monopoly", False):
            opp_mon[pr.owner] += 1
    unowned = sum(1 for sq, pr in env.properties.items() if pr.owner is None)
    rnd = float(getattr(env, "round", 0))
    # exact build shortfall (the switch between build-funding and war-chest mortgaging)
    shortfall = 0.0
    reserve = float(getattr(ctx, "reserve", 0.0))
    for group in GROUP_ORDER:
        squares = _GROUP_SQUARES[group]
        props = [env.properties[sq] for sq in squares]
        if any(pr.owner != pid for pr in props) or not getattr(props[0], "is_monopoly", False):
            continue
        levels = [pr.houses for pr in props]
        floor_level = min(levels)
        if floor_level >= 3:
            continue
        lump = sum(1 for lv in levels if lv == floor_level) * _HOUSE_PRICE[squares[0]]
        need = lump + reserve - cash
        if need > 0:
            shortfall = max(shortfall, need)
    debt = float(getattr(env, "debt_amount", 0.0) or 0.0)
    own_debt = debt if int(getattr(env, "debt_player", -1) or -1) == pid else 0.0
    chest_active = 1.0 if (shortfall <= 0 and unowned > 0 and cash < chest_target) else 0.0
    return np.array([
        min(cash, 6000.0) / 1500.0,
        rnd / 200.0,
        min(float(p.net_worth()), 20000.0) / 5000.0,
        n_owned / 28.0,
        own_mon / 4.0,
        max(opp_mon) / 4.0,
        reserve / 500.0,
        max(-2.0, min(4.0, (chest_target - cash) / 1000.0)),
        min(liq, 10000.0) / 1000.0,
        unowned / 28.0,
        1.0 if rnd < 25 else 0.0,
        1.0 if 25 <= rnd < 70 else 0.0,
        min(float(getattr(ctx, "danger_per_turn", lambda: 0.0)() if callable(getattr(ctx, "danger_per_turn", None)) else 0.0), 500.0) / 100.0,
        min(own_debt, 2000.0) / 500.0,
        min(shortfall, 2000.0) / 500.0,
        chest_active,
    ], dtype=np.float32)


def featurize(env, pid, cand):
    """(len(cand), FEAT_DIM) rows. Per-square ingredients computed once, indexed per candidate."""
    ctx = fl.make_context(env, pid)
    try:
        dead = set(fl.dead_deeds(ctx))
    except Exception:
        dead = set()
    try:
        chest = float(fl.chest_target(ctx))
    except Exception:
        chest = 0.0
    p = env.players[pid]
    cash = max(1.0, float(getattr(p, "money", getattr(p, "cash", 1.0))))

    # per-square tables (28 squares x things), cached by ctx._deed_cache underneath
    dv_self = np.zeros(40, dtype=np.float32)
    dv_opp = np.zeros(40, dtype=np.float32)
    comp_self = np.zeros(40, dtype=np.float32)
    comp_opp = np.zeros(40, dtype=np.float32)
    sq_inc = np.zeros(40, dtype=np.float32)
    blk_inc = np.zeros(40, dtype=np.float32)
    price = np.ones(40, dtype=np.float32)
    opps = [o for o in range(4) if o != pid and not env.players[o].bankrupt]
    for sq in PROPERTY_IDS:
        prop = env.properties.get(int(sq))
        if prop is None:
            continue
        price[sq] = max(1.0, float(prop.price))
        try:
            dv_self[sq] = ctx.deed_value(int(sq), pid)
            dv_opp[sq] = max((ctx.deed_value(int(sq), o) for o in opps), default=0.0)
            comp_self[sq] = 1.0 if ctx.completes_for(int(sq), pid) else 0.0
            comp_opp[sq] = 1.0 if any(ctx.completes_for(int(sq), o) for o in opps) else 0.0
            sq_inc[sq] = ctx.square_income(int(sq))
            blk_inc[sq] = ctx.block_income(int(sq))
        except Exception:
            pass

    sb = state_block(env, pid, ctx, chest)
    out = np.zeros((cand.size, FEAT_DIM), dtype=np.float32)
    out[:, :N_ACT_F] = action_features(env, pid, cand)[:, :N_ACT_F]
    out[:, N_ACT_F:N_ACT_F + N_STATE_F] = sb

    st = _STATIC[cand]
    sec, local, pos = st[:, 0], st[:, 1], st[:, 2]
    base = N_ACT_F + N_STATE_F
    for i in range(cand.size):
        s, l, q = int(sec[i]), int(local[i]), int(pos[i])
        name = SEC_NAMES[s]
        row = out[i, base:]
        row[14] = min(cand.size, 60) / 60.0
        row[15] = l / max(1.0, float(_STATIC[:, 0].tolist().count(s)))
        if q >= 0:
            row[0] = min(dv_self[q] / price[q], 8.0)
            row[1] = min(dv_self[q], 4000.0) / 500.0
            row[2] = comp_self[q]
            row[3] = comp_opp[q]
            row[4] = min(sq_inc[q], 500.0) / 50.0
            row[5] = min(blk_inc[q], 2000.0) / 100.0
            row[6] = 1.0 if q in dead else 0.0
            row[7] = min(price[q] / cash, 8.0)
            row[8] = min(dv_opp[q] / price[q], 8.0)
            prop_q = env.properties.get(q)
            mv = float(getattr(prop_q, "mortgage_value", price[q] // 2)) if prop_q is not None else price[q] / 2.0
            inc_um = 0.0
            try:
                inc_um = ctx.square_income(q, assume_unmortgaged=True)
            except Exception:
                pass
            row[22] = min(mv / cash, 8.0)
            row[23] = min(inc_um / max(mv, 1.0), 0.5) * 20.0
            try:
                row[24] = 1.0 if ctx.is_development_group(q) else 0.0
            except Exception:
                pass
            buf = 250.0  # redeem buffer, frozen with the other engineered constants
            row[25] = min(1.1 * mv / cash, 8.0)
            row[26] = max(-4.0, min(4.0, (cash - buf - 1.1 * mv) / 500.0))
            row[28] = min(mv, 400.0) / 400.0
            cost = 1.1 * mv
            mono_mult = 3.0 if (prop_q is not None and getattr(prop_q, "is_monopoly", False)) else 1.0
            row[31] = min(inc_um * mono_mult / max(cost, 1.0), 0.5) * 20.0
            row[32] = max(-4.0, min(4.0, (cash - cost - float(getattr(ctx, "reserve", 0.0)) - 250.0) / 500.0))
            row[33] = row[6] if getattr(ctx, "unowned_deeds", 0) > 0 else 0.0
            if name in ("improve_house", "improve_hotel"):
                colour = _COLOR_OF.get(q)
                if colour in GROUP_STEP_EFFICIENCY:
                    lvl_now = int(env.properties[q].houses)
                    nxt = min(lvl_now + 1, 5)
                    row[29] = min(GROUP_STEP_EFFICIENCY[colour][nxt], 200.0) / 50.0
                    squares = _GROUP_SQUARES[colour]
                    levels = [env.properties[s].houses for s in squares]
                    lump = sum(1 for lv in levels if lv == min(levels)) * _HOUSE_PRICE[squares[0]]
                    row[30] = 1.0 if cash >= lump + float(getattr(ctx, "reserve", 0.0)) else 0.0
        if name in ("buy_trade", "sell_trade", "exch_trade"):
            cp_rel, main, req, lvl_i = _decode_trade(int(cand[i]))
            others_abs = [x for x in range(4) if x != pid]
            cp = others_abs[cp_rel] if 0 <= cp_rel < len(others_abs) else -1
            if cp >= 0 and not env.players[cp].bankrupt:
                cp_cash = max(1.0, float(getattr(env.players[cp], "money",
                                                 getattr(env.players[cp], "cash", 1.0))))
                row[16] = min(cp_cash, 6000.0) / 1500.0
                row[21] = (cp_rel + 1) / 3.0
                if main >= 0:
                    try:
                        row[17] = 1.0 if ctx.completes_for(int(main), cp) else 0.0
                        row[18] = min(ctx.deed_value(int(main), cp) / price[main], 8.0)
                    except Exception:
                        pass
                if req >= 0:
                    try:
                        dv_cp_req = ctx.deed_value(int(req), cp)
                        row[19] = min(dv_cp_req / price[req], 8.0)
                        row[20] = min(dv_self[req] / price[req], 8.0)
                        if main >= 0:
                            dv_cp_main = ctx.deed_value(int(main), cp)
                            row[27] = max(-4.0, min(4.0, (dv_cp_main - dv_cp_req) / 250.0))
                    except Exception:
                        pass
            if lvl_i >= 0 and main >= 0:
                lvl = (0.75, 1.0, 1.25)[lvl_i]
                row[9] = lvl
                row[10] = min(lvl * price[main] / cash, 8.0)
            if name == "exch_trade" and main >= 0 and req >= 0:
                row[11] = min(dv_self[req] / price[req], 8.0)
                row[12] = max(-4.0, min(4.0, (dv_self[req] - dv_self[main]) / 250.0))
                pr1, pr2 = env.properties.get(int(main)), env.properties.get(int(req))
                row[13] = 1.0 if (pr1 is not None and pr2 is not None
                                  and getattr(pr1, "color", 0) == getattr(pr2, "color", 1)) else 0.0
    return out
