"""
configs_ntbn.py: Theoretical and Empirical Suitability Audit of 14 Benchmark Configurations for NTBN.
(AuditFix Edition: Rigorous mathematical justifications without overgeneralization).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass(frozen=True)
class NTBNConfigSuitability:
    index: int
    group: str
    variant: str
    cost_s: float
    n_inst: int
    payoff_type: str
    is_applicable: bool
    suitability_category: str
    justification: str


def audit_14_configs_suitability() -> List[NTBNConfigSuitability]:
    """
    Evaluates each of the 14 audit configurations against NTBN theoretical foundations:
    
    Theoretical Assumptions in Imaki et al. (2021) [Proposition 4.1]:
    1. Transaction Cost: Proportional cost c > 0.
       - Under c == 0 (frictionless market), there is no penalty for continuous rebalancing.
       - For smooth payoffs (e.g. European call/put under Black-Scholes), the theoretical hedge is delta*.
       - For non-smooth payoffs (barrier, lookback, digital) in discrete time, optimal hedge may diverge
         from continuous Delta, but an 'indifference no-transaction band' is not defined because trading is costless.
       - We DO NOT arbitrarily add transaction costs where the original configuration had zero cost.
    2. Payoff Smoothness: Proposition 4.1 assumes derivative payoff Z is sufficiently smooth so that
       loss function is twice differentiable with respect to S_t. For barrier options, payoff is discontinuous,
       violating this assumption.
    3. Dimensionality: Proposition 4.1 is proven for 1 hedging instrument (n_inst = 1).
       Multi-instrument settings (n_inst > 1) involve multi-dimensional convex cones/polytopes.
    """
    from experiments.quality_audit_14configs.configs import AUDIT_CONFIGS
    
    results = []
    for ac in AUDIT_CONFIGS:
        cost = ac.cost_s
        n_inst = ac.n_inst
        payoff = ac.payoff_type
        
        if cost > 0.0 and n_inst == 1:
            cat = "HOÀN HẢO (Textbook Match)"
            app = True
            just = f"Chi phí c={cost} > 0 và n_inst=1. Thỏa mãn đầy đủ các giả định của Định lý 4.1 (Imaki et al., 2021)."
        elif cost > 0.0 and n_inst > 1:
            cat = "ÁP DỤNG ĐƯỢC CÓ ĐIỀU KIỆN (Multi-Instrument Separable Bands)"
            app = True
            just = (f"Chi phí c_s={cost} > 0 nhưng có {n_inst} công cụ phòng hộ. Áp dụng được dưới dạng dải phân tách độc lập "
                    f"([b_l^(1), b_u^(1)] x [b_l^(2), b_u^(2)]), nhưng chưa tối ưu hình học đa diện tương quan (Imaki p.14).")
        else: # cost == 0
            if "barrier" in payoff or "digital" in payoff:
                cat = "KHÔNG PHÙ HỢP (Zero Cost & Non-smooth Payoff)"
                app = False
                just = (f"Cấu hình gốc có cost_s = 0.0 và payoff không trơn ({payoff}). "
                        "Vi phạm cả giả định chi phí dương c > 0 lẫn tính khả vi bậc 2 của Proposition 4.1. "
                        "Không áp dụng NTBN và không gán dải bằng Black-Scholes Delta.")
            else:
                cat = "KHÔNG PHÙ HỢP (Zero Friction Cost)"
                app = False
                just = (f"Cấu hình gốc có cost_s = 0.0 (thị trường không ma sát). "
                        "Theo lý thuyết, không có vùng bàng quan giao dịch vì giao dịch là miễn phí. "
                        "Không tự ý thêm chi phí giao dịch vào cấu hình này.")
            
        results.append(NTBNConfigSuitability(
            index=ac.index,
            group=ac.group,
            variant=ac.variant,
            cost_s=cost,
            n_inst=n_inst,
            payoff_type=payoff,
            is_applicable=app,
            suitability_category=cat,
            justification=just
        ))
    return results


def print_suitability_report():
    audits = audit_14_configs_suitability()
    print("=" * 115)
    print(f"{'Idx':<4} | {'Group':<18} | {'Variant':<20} | {'Cost':<8} | {'Inst':<5} | {'Phù hợp?':<10} | {'Phân loại'}")
    print("-" * 115)
    for a in audits:
        app_str = "CÓ" if a.is_applicable else "KHÔNG"
        print(f"{a.index:<4} | {a.group:<18} | {a.variant:<20} | {a.cost_s:<8.4f} | {a.n_inst:<5} | {app_str:<10} | {a.suitability_category}")
    print("=" * 115)


if __name__ == "__main__":
    print_suitability_report()
