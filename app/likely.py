"""最可能诱因（MPE, most probable explanation）：在已构建的 ROBDD 上
求使条件函数成立、独立先验乘积最大的完整事件真假赋值。

变量顺序即事件 id 字典序（var 0 最靠前）。一个完整事件赋值 A 的权重为
    P(A) = Π_v  p_v       （A 中 v=true）
                (1-p_v)   （A 中 v=false）。
目标：在 root 表示的条件 C 成立的全部完整赋值中取 P(A) 最大者；权重
并列时按 (var0, var1, …) 的真假序列取字典序（false < true）。

自底向上对唯一表做一遍动态规划：
  W(n) = 在“到达节点 n”的前提下，从 n 的变量等级到最后一个变量
         （含被 BDD 边跳过的自由变量）的最优权重乘积；
  M(n) = 对应最优完整后缀的位掩码（位 i 置 1 表示事件 i 取 true）。
ROBDD 边跳过的变量在该子图中自由，逐位独立取先验因子较大的一侧，
恰为 1/2 时按并列规则取 false。节点处 low（当前位取 false）与
high（当前位取 true）两支比较最优权重：等权时首个分歧位正是当前
变量，low 支在全局字典序中必然占先，故仅严格更优才替换。

全程只遍历唯一表一次，节点数与事件数均为线性规模，不枚举事件状态。
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Tuple

from .bdd import ROBDD

# (最优完整赋值的先验乘积 P(A*), 完整赋值位掩码)；条件不可满足时为 None
Explanation = Optional[Tuple[Fraction, int]]


def best_explanation(
    bdd: ROBDD, root: int, probs: List[Fraction]
) -> Explanation:
    """求使 root=1 的先验乘积最大的完整事件赋值。

    返回 (最大权重 P(A*), 赋值位掩码)；root 恒假，或先验下不存在
    正概率的满足赋值时返回 None。
    """
    if root == 0:
        return None

    n = bdd.n_vars

    # 每个变量“自由”（不出现在所经子函数中）时的最优取值：取先验
    # 因子较大的一侧；p == 1-p == 1/2 时按并列规则取 false
    free_factor: List[Fraction] = [Fraction(0)] * n
    free_bit: List[int] = [0] * n
    for i in range(n):
        p = probs[i]
        if p > 1 - p:
            free_factor[i], free_bit[i] = p, 1
        else:
            free_factor[i], free_bit[i] = 1 - p, 0

    def free_segment(lo: int, hi: int) -> Tuple[Fraction, int]:
        """变量区间 [lo, hi) 全部自由时的最优因子乘积与取值掩码。"""
        weight = Fraction(1)
        mask = 0
        for k in range(lo, hi):
            weight *= free_factor[k]
            mask |= free_bit[k] << k
        return weight, mask

    if root == 1:
        return free_segment(0, n)

    # 终端 0 不可满足（权重 0）；终端 1 已无待决变量（权重 1，掩码 0）。
    # dp_w[nid]：节点 nid 子图（其变量等级及之后，含被边跳过的自由
    # 变量）上使函数为 1 的最优权重；dp_m[nid] 为对应赋值后缀位掩码。
    dp_w: List[Fraction] = [Fraction(0)] * len(bdd.nodes)
    dp_m: List[int] = [0] * len(bdd.nodes)
    dp_w[1] = Fraction(1)

    # 子节点 id 总是小于父节点 id，递增序即为后序遍历
    for nid in range(2, len(bdd.nodes)):
        node = bdd.nodes[nid]
        assert node is not None
        var, low, high = node

        candidates: List[Tuple[Fraction, int]] = []

        # low 支：var=false，乘 (1-p_var)；边到子节点之间跳过的变量自由
        if dp_w[low] > 0:
            child_var = bdd.nodes[low][0] if low >= 2 else n
            seg_w, seg_m = free_segment(var + 1, child_var)
            candidates.append((
                (1 - probs[var]) * seg_w * dp_w[low],
                seg_m | dp_m[low],
            ))

        # high 支：var=true，乘 p_var
        if dp_w[high] > 0:
            child_var = bdd.nodes[high][0] if high >= 2 else n
            seg_w, seg_m = free_segment(var + 1, child_var)
            candidates.append((
                probs[var] * seg_w * dp_w[high],
                (1 << var) | seg_m | dp_m[high],
            ))

        if candidates:
            # low 支排在首位；high 支仅在严格更优时替换——等权时
            # low 支当前位为 false，全局字典序必然占先
            best_w, best_m = candidates[0]
            for cand_w, cand_m in candidates[1:]:
                if cand_w > best_w:
                    best_w, best_m = cand_w, cand_m
            dp_w[nid], dp_m[nid] = best_w, best_m

    if dp_w[root] == 0:
        return None

    # root 变量之前的变量在整条函数上自由
    root_var = bdd.nodes[root][0]  # type: ignore[index]
    head_w, head_m = free_segment(0, root_var)
    return head_w * dp_w[root], head_m | dp_m[root]
