"""共享事件故障概率分析 API。

POST /v1/analyze
  输入：events（1~32 个独立事件，概率为 n/d 分数字符串）、
        gates（0~64 个布尔门 AND/OR/NOT）、顶层节点 top。
  输出：顶层发生概率与每个事件的 Birnbaum 重要度，
        P(top|事件=1) − P(top|事件=0)，均为约分分数字符串。

POST /v1/cut-sets
  输入格式与校验同 /v1/analyze（p 仍须合法，但不参与割集计算），
  仅接受 AND/OR 门（出现 NOT 返回 422）。
  输出：顶事件的全部最小割集；任一门去重、吸收后仍超过 5000 组
        时返回 422。

POST /v1/repair-plan
  另收覆盖全部事件 id 的 states（事件当前布尔状态）与
  repair_costs（1~10^6 的整数修复费用），其余输入与校验同上。
  只能把当前为 true 的事件修复为 false，共享引用共用一次修复；
  求令顶事件变为 false 的最优方案，依次按总成本、修复数、排序后的
  事件 id 序列取最小。顶事件原本为 false 时取空方案、零成本；
  无解时返回 possible=false。

POST /v1/diagnose
  另收 observations：1~16 个已声明事件或门 id 到严格布尔值的映射，
  表示同一时刻的读数。令 E 为全部读数同时成立；不同事件仍按原概率
  独立，同一事件的引用共享状态，AND/OR/NOT 均可观测。
  返回 evidence_probability=P(E) 与 posteriors（每个事件 id 的
  P(事件=1 且 E)/P(E)），均为约分 n/d 字符串。观测是条件证据，
  门读数不被当作相互独立的事件，也不修改先验；P(E)=0 返回 422。
  同一 ROBDD 合取观测约束后精确加权，不枚举事件状态；累计非终端
  节点超过 50000 或观测键/值非法时返回 422。

POST /v1/inspect-next
  另收 observations（0~16 个，可为空，格式同 /v1/diagnose）与
  candidates（1~16 个互异、已声明且未被观测的事件或门 id）。
  令 E 为已有读数的合取；对每个候选 X 求 E 下 X=true/false 的概率
  与对应分支的顶事件后验 q（零概率分支的 q 为 null），并以
  G(q)=2q(1-q) 衡量顶事件不确定性：收益 = G(P(top|E)) − 两分支按
  概率加权的 G(q)。返回按 id 排序的各候选分支概率、q 与收益，
  按收益最大、id 最小给出 recommended_id，分数均约分。同样复用
  ROBDD 精确计算，不枚举事件状态；P(E)=0、候选非法或累计非终端
  节点超过 50000 时返回 422。

POST /v1/likely-cause
  另收 observations（0~16 个，可为空，格式同 /v1/diagnose）。
  令 C 为顶事件发生且全部读数成立；在满足 C 的完整事件真假赋值
  A 中，按各事件独立先验乘积选 P(A) 最大者，并列时按事件 id
  字典序比较真假序列、false 优先。返回按 id 排列的 assignment、
  condition_probability=P(C) 与 assignment_posterior=P(A)/P(C)，
  分数均约分。共享引用、NOT 门与未被结构约束的事件都参与赋值及
  概率计算；复用 ROBDD 精确求最优赋值，不枚举事件状态；P(C)=0
  或累计非终端节点超过 50000 时返回 422。

所有错误输入（未知引用、环、非法概率、节点超限等）统一返回 422。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, StrictInt
from pydantic.config import ConfigDict

from .bdd import MAX_NONTERMINAL_NODES, NodeLimitExceeded, ROBDD
from .cutsets import compute_cut_set_families
from .likely import most_likely_assignment
from .repair import best_repair

app = FastAPI(
    title="共享事件故障概率 API",
    description="基于 ROBDD 的精确故障树顶事件概率与事件重要度计算",
    version="1.0.0",
)

GateOp = Literal["AND", "OR", "NOT"]


class EventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="唯一 ASCII 事件标识")
    p: str = Field(..., description="发生概率，n/d 形式，0<=n<=d<=10^6")


class GateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="唯一 ASCII 门标识")
    op: GateOp = Field(..., description="门算子：AND / OR / NOT")
    inputs: List[str] = Field(..., description="输入引用，可为事件或门")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: List[EventIn] = Field(..., min_length=1, max_length=32)
    gates: List[GateIn] = Field(default_factory=list, max_length=64)
    top: str = Field(..., min_length=1, description="顶层节点 id（事件或门）")


class AnalyzeResponse(BaseModel):
    top_probability: str
    importances: Dict[str, str]


class CutSetsResponse(BaseModel):
    cut_sets: List[List[str]]


class RepairPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: List[EventIn] = Field(..., min_length=1, max_length=32)
    gates: List[GateIn] = Field(default_factory=list, max_length=64)
    top: str = Field(..., min_length=1, description="顶层节点 id（事件或门）")
    states: Dict[str, StrictBool] = Field(
        ..., description="每个事件的当前状态：true=发生，false=未发生"
    )
    repair_costs: Dict[str, StrictInt] = Field(
        ..., description="每个事件的修复费用，取值 1~10^6 的整数"
    )


class RepairPlanResponse(BaseModel):
    possible: bool
    repair_events: List[str]
    total_cost: Optional[int]


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: List[EventIn] = Field(..., min_length=1, max_length=32)
    gates: List[GateIn] = Field(default_factory=list, max_length=64)
    top: str = Field(..., min_length=1, description="顶层节点 id（事件或门）")
    observations: Dict[str, StrictBool] = Field(
        ...,
        min_length=1,
        max_length=16,
        description=(
            "同一时刻的读数：1~16 个已声明事件或门 id 到严格布尔值的映射"
        ),
    )


class DiagnoseResponse(BaseModel):
    evidence_probability: str
    posteriors: Dict[str, str]


class InspectNextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: List[EventIn] = Field(..., min_length=1, max_length=32)
    gates: List[GateIn] = Field(default_factory=list, max_length=64)
    top: str = Field(..., min_length=1, description="顶层节点 id（事件或门）")
    observations: Dict[str, StrictBool] = Field(
        default_factory=dict,
        max_length=16,
        description=(
            "已有读数：0~16 个已声明事件或门 id 到严格布尔值的映射，可为空"
        ),
    )
    candidates: List[str] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="待评估的候选：1~16 个互异、已声明且未被观测的事件或门 id",
    )


class InspectBranch(BaseModel):
    probability: str = Field(..., description="E 下该分支的概率，约分 n/d")
    q: Optional[str] = Field(
        ..., description="该分支下顶事件后验 P(top|E∧分支)；零概率分支为 null"
    )


class InspectCandidate(BaseModel):
    id: str
    branches: Dict[Literal["true", "false"], InspectBranch]
    gain: str = Field(..., description="观测该候选的收益，约分 n/d")


class InspectNextResponse(BaseModel):
    evidence_probability: str
    top_posterior: str = Field(..., description="P(top|E)，约分 n/d")
    candidates: List[InspectCandidate]
    recommended_id: str


class LikelyCauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: List[EventIn] = Field(..., min_length=1, max_length=32)
    gates: List[GateIn] = Field(default_factory=list, max_length=64)
    top: str = Field(..., min_length=1, description="顶层节点 id（事件或门）")
    observations: Dict[str, StrictBool] = Field(
        default_factory=dict,
        max_length=16,
        description=(
            "已有读数：0~16 个已声明事件或门 id 到严格布尔值的映射，可为空"
        ),
    )


class LikelyCauseResponse(BaseModel):
    assignment: Dict[str, StrictBool] = Field(
        ..., description="P(A) 最大的完整事件赋值，按事件 id 字典序排列"
    )
    condition_probability: str = Field(
        ..., description="P(C)：顶事件发生且全部读数成立的概率，约分 n/d"
    )
    assignment_posterior: str = Field(
        ..., description="P(A)/P(C)，约分 n/d"
    )


def _bad_request(reason: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": reason})


def _parse_probability(raw: str) -> Fraction:
    """解析 n/d 形式的概率：必须严格为非负整数/正整数，且 n<=d<=10^6。"""
    parts = raw.split("/")
    if len(parts) != 2:
        raise ValueError(f"非法概率 {raw!r}：必须为 n/d 形式")
    ns, ds = parts
    if not ns or not ds:
        raise ValueError(f"非法概率 {raw!r}：分子或分母为空")
    # 严格十进制整数：拒绝符号、空白、前导 + 等
    if not (ns.isdigit() and ds.isdigit()):
        raise ValueError(f"非法概率 {raw!r}：分子分母必须为非负十进制整数")
    n, d = int(ns), int(ds)
    if d == 0:
        raise ValueError(f"非法概率 {raw!r}：分母不能为 0")
    if n > d:
        raise ValueError(f"非法概率 {raw!r}：分子不能大于分母")
    if d > 1_000_000:
        raise ValueError(f"非法概率 {raw!r}：分母不能超过 10^6")
    return Fraction(n, d)


def _is_ascii_nonempty(s: str) -> bool:
    return len(s) > 0 and s.isascii()


def _validate_payload(
    payload: AnalyzeRequest,
) -> "tuple[Dict[str, Fraction], Dict[str, GateOp], Dict[str, List[str]]]":
    """事件与门的全部静态校验，返回 (事件概率表, 门算子表, 门输入表)。

    非法概率、id 冲突/重复、元数错误、未知引用、环等均抛 ValueError。
    """
    events = payload.events
    gates = payload.gates

    # ---- 事件：id、概率 ----
    probs: Dict[str, Fraction] = {}
    for ev in events:
        if not _is_ascii_nonempty(ev.id):
            raise ValueError(f"非法事件 id {ev.id!r}：必须为非空 ASCII 字符串")
        if ev.id in probs:
            raise ValueError(f"事件 id 重复：{ev.id!r}")
        probs[ev.id] = _parse_probability(ev.p)

    # ---- 门：id、算子输入元数、重复 ----
    gate_ops: Dict[str, GateOp] = {}
    gate_inputs: Dict[str, List[str]] = {}
    for gate in gates:
        if not _is_ascii_nonempty(gate.id):
            raise ValueError(f"非法门 id {gate.id!r}：必须为非空 ASCII 字符串")
        if gate.id in probs:
            raise ValueError(f"门 id 与事件 id 冲突：{gate.id!r}")
        if gate.id in gate_ops:
            raise ValueError(f"门 id 重复：{gate.id!r}")
        n_inputs = len(gate.inputs)
        if gate.op == "NOT":
            if n_inputs != 1:
                raise ValueError(
                    f"门 {gate.id!r} 为 NOT，必须恰好有 1 个输入，实际 {n_inputs} 个"
                )
        elif n_inputs < 2 or n_inputs > 8:
            raise ValueError(
                f"门 {gate.id!r} 为 {gate.op}，必须有 2~8 个输入，实际 {n_inputs} 个"
            )
        gate_ops[gate.id] = gate.op
        gate_inputs[gate.id] = list(gate.inputs)

    known_ids = set(probs) | set(gate_ops)

    # ---- 引用完整性（门输入与顶层节点）----
    for gid, inputs in gate_inputs.items():
        for ref in inputs:
            if ref not in known_ids:
                raise ValueError(f"门 {gid!r} 引用了未知节点：{ref!r}")
    if payload.top not in known_ids:
        raise ValueError(f"顶层节点引用了未知 id：{payload.top!r}")

    # ---- 环检测：门依赖图上的 DFS 三色标记 ----
    # 0=未访问, 1=在当前递归栈中, 2=已完成
    color: Dict[str, int] = {gid: 0 for gid in gate_ops}

    def dfs(node_id: str) -> None:
        if node_id not in color:  # 事件节点是叶子
            return
        if color[node_id] == 1:
            raise ValueError(f"门的引用关系存在环（涉及门 {node_id!r}）")
        if color[node_id] == 2:
            return
        color[node_id] = 1
        for child in gate_inputs[node_id]:
            dfs(child)
        color[node_id] = 2

    for gid in gate_ops:
        dfs(gid)

    return probs, gate_ops, gate_inputs


def build_tree_robdd(
    probs: Dict[str, Fraction],
    gate_ops: Dict[str, GateOp],
    gate_inputs: Dict[str, List[str]],
    top: str,
    also: Tuple[str, ...] = (),
) -> Tuple[List[str], ROBDD, int, Dict[str, int]]:
    """按事件 id 字典序为变量顺序构建故障树的 ROBDD。

    返回 (字典序事件 id 列表, ROBDD, 顶层节点 id, 节点表)。同一事件
    的多处引用共用同一个基础变量节点；NOT 门以 BDD 取反求值。
    also 中列出的额外门 id（必须已通过引用/环校验）也一并求值并
    收入节点表，供观测等场景按门 id 取其 BDD 节点。
    """
    sorted_events = sorted(probs)
    var_index = {eid: i for i, eid in enumerate(sorted_events)}

    bdd = ROBDD(
        n_vars=len(sorted_events), limit=MAX_NONTERMINAL_NODES
    )

    # 事件变量对应只含自身判定的基础节点
    event_nodes = {
        eid: bdd.mk(var_index[eid], 0, 1) for eid in sorted_events
    }

    built: Dict[str, int] = {}

    def eval_node(node_id: str) -> int:
        if node_id in event_nodes:
            return event_nodes[node_id]
        if node_id in built:
            return built[node_id]
        op = gate_ops[node_id]
        inputs = gate_inputs[node_id]
        if op == "NOT":
            result = bdd.negate(eval_node(inputs[0]))
        else:
            result = eval_node(inputs[0])
            for ref in inputs[1:]:
                nxt = eval_node(ref)
                if op == "AND":
                    result = bdd.and_(result, nxt)
                else:
                    result = bdd.or_(result, nxt)
        built[node_id] = result
        return result

    top_node = eval_node(top)
    for node_id in also:
        eval_node(node_id)
    node_map: Dict[str, int] = {**event_nodes, **built}
    return sorted_events, bdd, top_node, node_map


def analyze_logic(payload: AnalyzeRequest) -> AnalyzeResponse:
    probs, gate_ops, gate_inputs = _validate_payload(payload)

    sorted_events, bdd, top_node, _ = build_tree_robdd(
        probs, gate_ops, gate_inputs, payload.top
    )
    prob_list = [probs[eid] for eid in sorted_events]
    top_prob, importance = bdd.root_probability(top_node, prob_list)

    def frac_string(value: Fraction) -> str:
        # 统一输出约分后的 n/d（整数时分母为 1）
        return f"{value.numerator}/{value.denominator}"

    return AnalyzeResponse(
        top_probability=frac_string(top_prob),
        importances={eid: frac_string(importance[i])
                     for i, eid in enumerate(sorted_events)},
    )


def cut_sets_logic(payload: AnalyzeRequest) -> CutSetsResponse:
    probs, gate_ops, gate_inputs = _validate_payload(payload)

    # 割集仅适用于单调结构：NOT 会破坏“组内事件同时发生则顶事件
    # 必然发生”的语义，凡出现 NOT 门一律拒绝
    for gid, op in gate_ops.items():
        if op == "NOT":
            raise ValueError(
                f"门 {gid!r} 为 NOT：最小割集仅支持 AND/OR 单调结构，"
                f"含取反的概率分析请使用 /v1/analyze"
            )

    sorted_events = sorted(probs)
    families = compute_cut_set_families(sorted_events, gate_ops, gate_inputs)
    if payload.top in probs:
        top_masks = {1 << sorted_events.index(payload.top)}
    else:
        top_masks = families[payload.top]

    # 位序即事件 id 字典序，展开后组内天然有序
    cut_lists = [
        [eid for i, eid in enumerate(sorted_events) if mask >> i & 1]
        for mask in top_masks
    ]
    # 组间：先按割集长度，再按事件 id 字典序
    cut_lists.sort(key=lambda ids: (len(ids), ids))
    return CutSetsResponse(cut_sets=cut_lists)


def _check_cover(
    values: Dict[str, object], event_ids: List[str], field: str
) -> List[str]:
    """检查 field 的键集合恰好覆盖全部事件 id，返回收集到的错误信息。"""
    errors: List[str] = []
    missing = [eid for eid in event_ids if eid not in values]
    extra = [k for k in values if k not in event_ids]
    if missing:
        errors.append(
            f"{field} 缺少事件的对应项：{', '.join(repr(e) for e in missing)}"
        )
    if extra:
        errors.append(
            f"{field} 出现了不属于任何事件的键："
            f"{', '.join(repr(e) for e in extra)}"
        )
    return errors


def repair_plan_logic(payload: RepairPlanRequest) -> RepairPlanResponse:
    probs, gate_ops, gate_inputs = _validate_payload(payload)
    sorted_events = sorted(probs)

    # states 与 repair_costs 必须恰好覆盖全部事件 id；多项错误一并返回
    errors = _check_cover(payload.states, sorted_events, "states")
    errors += _check_cover(
        payload.repair_costs, sorted_events, "repair_costs"
    )
    if errors:
        raise ValueError("；".join(errors))

    # 修复费用：1~10^6 的整数
    bad_costs = [
        f"{eid!r}={payload.repair_costs[eid]}"
        for eid in sorted_events
        if not 1 <= payload.repair_costs[eid] <= 1_000_000
    ]
    if bad_costs:
        raise ValueError(
            "repair_costs 的取值必须为 1~10^6 的整数，非法项："
            + ", ".join(bad_costs)
        )

    sorted_events, bdd, top_node, _ = build_tree_robdd(
        probs, gate_ops, gate_inputs, payload.top
    )

    fixed_false_mask = 0
    for i, eid in enumerate(sorted_events):
        if payload.states[eid] is False:
            fixed_false_mask |= 1 << i
    costs = [payload.repair_costs[eid] for eid in sorted_events]

    plan = best_repair(bdd, top_node, fixed_false_mask, costs)
    if plan is None:
        return RepairPlanResponse(
            possible=False, repair_events=[], total_cost=None
        )
    total_cost, mask = plan
    repair_events = [
        eid for i, eid in enumerate(sorted_events) if mask >> i & 1
    ]
    return RepairPlanResponse(
        possible=True, repair_events=repair_events, total_cost=total_cost
    )


def diagnose_logic(payload: DiagnoseRequest) -> DiagnoseResponse:
    probs, gate_ops, gate_inputs = _validate_payload(payload)
    sorted_events = sorted(probs)

    # 观测键必须指向已声明的事件或门（门读数只是结构约束，不是相互
    # 独立的新事件；非法键多项时一并报错）
    known_ids = set(probs) | set(gate_ops)
    unknown = [k for k in payload.observations if k not in known_ids]
    if unknown:
        raise ValueError(
            "observations 引用了未声明的事件或门 id："
            + ", ".join(repr(k) for k in unknown)
        )

    # 被观测的门（即便 top 不可达）也要构建到同一 ROBDD 中
    observed_gates = tuple(
        k for k in payload.observations if k in gate_ops
    )
    _, bdd, _, node_map = build_tree_robdd(
        probs, gate_ops, gate_inputs, payload.top, also=observed_gates
    )

    # 在同一 ROBDD 上把各观测约束（node == 布尔读数）合取：
    # 读 1 即门/事件本身，读 0 即其 BDD 取反；同一事件/门的多处
    # 引用共享同一 BDD 节点，天然共享状态
    evidence = 1
    for obs_id, value in payload.observations.items():
        lit = node_map[obs_id]
        constraint = lit if value else bdd.negate(lit)
        evidence = bdd.and_(evidence, constraint)
        if evidence == 0:
            # 短路：结构上已不可同时满足，无需继续合取
            break

    prob_list = [probs[eid] for eid in sorted_events]
    evidence_prob, joint = bdd.joint_probabilities(evidence, prob_list)

    if evidence_prob == 0:
        if evidence == 0:
            raise ValueError(
                "观测相互矛盾：给定的事件/门读数不可能在同一时刻"
                "同时成立（结构约束的合取恒为假），evidence_probability=0"
            )
        raise ValueError(
            "观测的联合概率为 0：尽管结构上可满足，但按各事件的独立"
            "先验概率，任一使全部读数成立的事件状态概率都为 0"
            "（例如读数要求概率为 0 的事件发生），evidence_probability=0"
        )

    def frac_string(value: Fraction) -> str:
        return f"{value.numerator}/{value.denominator}"

    return DiagnoseResponse(
        evidence_probability=frac_string(evidence_prob),
        posteriors={
            eid: frac_string(joint[i] / evidence_prob)
            for i, eid in enumerate(sorted_events)
        },
    )


def inspect_next_logic(payload: InspectNextRequest) -> InspectNextResponse:
    probs, gate_ops, gate_inputs = _validate_payload(payload)
    sorted_events = sorted(probs)
    known_ids = set(probs) | set(gate_ops)

    # 观测键必须指向已声明的事件或门（同 /v1/diagnose）
    unknown = [k for k in payload.observations if k not in known_ids]
    if unknown:
        raise ValueError(
            "observations 引用了未声明的事件或门 id："
            + ", ".join(repr(k) for k in unknown)
        )

    # 候选校验：互异、已声明、未被观测；多项问题一并返回
    cand_errors: List[str] = []
    seen: set = set()
    dup: List[str] = []
    for cid in payload.candidates:
        if cid in seen and cid not in dup:
            dup.append(cid)
        seen.add(cid)
    if dup:
        cand_errors.append(
            "candidates 中存在重复的 id："
            + ", ".join(repr(c) for c in dup)
        )
    unknown_c = list(
        dict.fromkeys(c for c in payload.candidates if c not in known_ids)
    )
    if unknown_c:
        cand_errors.append(
            "candidates 引用了未声明的事件或门 id："
            + ", ".join(repr(c) for c in unknown_c)
        )
    observed_c = list(
        dict.fromkeys(
            c for c in payload.candidates if c in payload.observations
        )
    )
    if observed_c:
        cand_errors.append(
            "candidates 必须未被观测，以下 id 已出现在 observations 中："
            + ", ".join(repr(c) for c in observed_c)
        )
    if cand_errors:
        raise ValueError("；".join(cand_errors))

    # 被观测/被候选的门（即便 top 不可达）也构建到同一 ROBDD 中
    extra_gates = tuple(
        k
        for k in list(payload.observations) + list(dict.fromkeys(payload.candidates))
        if k in gate_ops
    )
    _, bdd, top_node, node_map = build_tree_robdd(
        probs, gate_ops, gate_inputs, payload.top, also=extra_gates
    )

    # 证据 E：各观测约束的合取（同 /v1/diagnose）
    evidence = 1
    for obs_id, value in payload.observations.items():
        lit = node_map[obs_id]
        constraint = lit if value else bdd.negate(lit)
        evidence = bdd.and_(evidence, constraint)
        if evidence == 0:
            break

    prob_list = [probs[eid] for eid in sorted_events]
    node_prob = bdd.node_probabilities(prob_list)
    evidence_prob = node_prob[evidence]

    if evidence_prob == 0:
        if evidence == 0:
            raise ValueError(
                "观测相互矛盾：给定的事件/门读数不可能在同一时刻"
                "同时成立（结构约束的合取恒为假），evidence_probability=0"
            )
        raise ValueError(
            "观测的联合概率为 0：尽管结构上可满足，但按各事件的独立"
            "先验概率，任一使全部读数成立的事件状态概率都为 0"
            "（例如读数要求概率为 0 的事件发生），evidence_probability=0"
        )

    # 先把全部分支合取构建完，再一次遍历读出所有根的概率：
    #   E∧top、每个候选的 E∧X 与 E∧X∧top；
    # 假分支用减法精确得到：P(E∧¬X)=P(E)−P(E∧X)，
    #   P(E∧¬X∧top)=P(E∧top)−P(E∧X∧top)，无需再建取反合取
    top_and_e = bdd.and_(top_node, evidence)
    branch_nodes: Dict[str, Tuple[int, int]] = {}
    for cid in dict.fromkeys(payload.candidates):
        e_and_x = bdd.and_(evidence, node_map[cid])
        e_and_x_top = bdd.and_(e_and_x, top_node)
        branch_nodes[cid] = (e_and_x, e_and_x_top)
    node_prob = bdd.node_probabilities(prob_list)

    top_and_e_prob = node_prob[top_and_e]
    q_e = top_and_e_prob / evidence_prob  # P(top|E)

    def g_uncertainty(q: Fraction) -> Fraction:
        return 2 * q * (1 - q)

    def frac_string(value: Fraction) -> str:
        return f"{value.numerator}/{value.denominator}"

    results: List[InspectCandidate] = []
    for cid in sorted(branch_nodes):
        e_and_x, e_and_x_top = branch_nodes[cid]
        p_ex = node_prob[e_and_x]          # P(E ∧ X)
        p_ext = node_prob[e_and_x_top]     # P(E ∧ X ∧ top)
        p_true = p_ex / evidence_prob
        p_false = 1 - p_true
        # 零概率分支的后验无定义，记 null；其加权收益贡献为 0
        q_true = (p_ext / p_ex) if p_ex > 0 else None
        rest = evidence_prob - p_ex        # P(E ∧ ¬X)
        q_false = (
            (top_and_e_prob - p_ext) / rest if rest > 0 else None
        )
        gain = g_uncertainty(q_e)
        if q_true is not None:
            gain -= p_true * g_uncertainty(q_true)
        if q_false is not None:
            gain -= p_false * g_uncertainty(q_false)
        results.append(
            InspectCandidate(
                id=cid,
                branches={
                    "true": InspectBranch(
                        probability=frac_string(p_true),
                        q=frac_string(q_true) if q_true is not None else None,
                    ),
                    "false": InspectBranch(
                        probability=frac_string(p_false),
                        q=frac_string(q_false)
                        if q_false is not None
                        else None,
                    ),
                },
                gain=frac_string(gain),
            )
        )

    # 收益最大者优先，并列取 id 最小（results 已按 id 升序，严格更大才替换）
    best = results[0]
    best_gain = Fraction(best.gain)
    for cand in results[1:]:
        cand_gain = Fraction(cand.gain)
        if cand_gain > best_gain:
            best, best_gain = cand, cand_gain

    return InspectNextResponse(
        evidence_probability=frac_string(evidence_prob),
        top_posterior=frac_string(q_e),
        candidates=results,
        recommended_id=best.id,
    )


def likely_cause_logic(payload: LikelyCauseRequest) -> LikelyCauseResponse:
    probs, gate_ops, gate_inputs = _validate_payload(payload)
    sorted_events = sorted(probs)

    # 观测键必须指向已声明的事件或门（同 /v1/diagnose）
    known_ids = set(probs) | set(gate_ops)
    unknown = [k for k in payload.observations if k not in known_ids]
    if unknown:
        raise ValueError(
            "observations 引用了未声明的事件或门 id："
            + ", ".join(repr(k) for k in unknown)
        )

    # 被观测的门（即便 top 不可达）也要构建到同一 ROBDD 中
    observed_gates = tuple(
        k for k in payload.observations if k in gate_ops
    )
    _, bdd, top_node, node_map = build_tree_robdd(
        probs, gate_ops, gate_inputs, payload.top, also=observed_gates
    )

    # 条件 C：顶事件发生且全部读数成立（同一 ROBDD 上的合取）
    condition = top_node
    for obs_id, value in payload.observations.items():
        lit = node_map[obs_id]
        constraint = lit if value else bdd.negate(lit)
        condition = bdd.and_(condition, constraint)
        if condition == 0:
            # 短路：结构上已不可同时满足，无需继续合取
            break

    prob_list = [probs[eid] for eid in sorted_events]
    node_prob = bdd.node_probabilities(prob_list)
    condition_prob = node_prob[condition]

    if condition_prob == 0:
        if condition == 0:
            raise ValueError(
                "顶事件与观测相互矛盾：顶事件发生且全部读数同时成立在"
                "结构上不可能（结构约束的合取恒为假），"
                "condition_probability=0"
            )
        raise ValueError(
            "顶事件发生且全部读数同时成立的联合概率为 0：尽管结构上"
            "可满足，但按各事件的独立先验概率，任一满足条件的事件状态"
            "概率都为 0（例如读数要求概率为 0 的事件发生），"
            "condition_probability=0"
        )

    # 在满足 C 的完整赋值中取独立先验乘积最大者（ROBDD 动态规划，
    # 不枚举事件状态）；未被结构约束的事件按自由变量取较可能一侧
    best_weight, assignment = most_likely_assignment(
        bdd, condition, prob_list
    )

    def frac_string(value: Fraction) -> str:
        return f"{value.numerator}/{value.denominator}"

    return LikelyCauseResponse(
        assignment={
            eid: assignment[i] for i, eid in enumerate(sorted_events)
        },
        condition_probability=frac_string(condition_prob),
        assignment_posterior=frac_string(best_weight / condition_prob),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """请求结构错误同样返回 422，并给出可直接阅读的中文原因。"""
    errs = exc.errors()
    messages = []
    for err in errs:
        loc = [p for p in err.get("loc", []) if p != "body"]
        where = ".".join(str(p) for p in loc) or "请求体"
        msg = err.get("msg", "参数无效")
        messages.append(f"{where}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": "；".join(messages)},
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return analyze_logic(payload)
    except NodeLimitExceeded as exc:
        # 构建中累计非终端节点超过限制
        return _bad_request(str(exc))  # type: ignore[return-value]
    except ValueError as exc:
        # 非法概率、未知引用、环、元数错误等全部语义错误
        return _bad_request(str(exc))  # type: ignore[return-value]


@app.post("/v1/cut-sets", response_model=CutSetsResponse)
async def cut_sets(payload: AnalyzeRequest) -> CutSetsResponse:
    try:
        return cut_sets_logic(payload)
    except ValueError as exc:
        # 非法概率、未知引用、环、NOT 门、割集超限等全部语义错误
        return _bad_request(str(exc))  # type: ignore[return-value]


@app.post("/v1/repair-plan", response_model=RepairPlanResponse)
async def repair_plan(payload: RepairPlanRequest) -> RepairPlanResponse:
    try:
        return repair_plan_logic(payload)
    except NodeLimitExceeded as exc:
        # 构建中累计非终端节点超过限制
        return _bad_request(str(exc))  # type: ignore[return-value]
    except ValueError as exc:
        # 未知引用、环、元数错误、状态/费用缺项、费用越界等全部语义错误
        return _bad_request(str(exc))  # type: ignore[return-value]


@app.post("/v1/diagnose", response_model=DiagnoseResponse)
async def diagnose(payload: DiagnoseRequest) -> DiagnoseResponse:
    try:
        return diagnose_logic(payload)
    except NodeLimitExceeded as exc:
        # 树构建或合取观测约束时累计非终端节点超过限制
        return _bad_request(str(exc))  # type: ignore[return-value]
    except ValueError as exc:
        # 未知引用、环、非法观测键、观测矛盾或联合概率为 0 等语义错误
        return _bad_request(str(exc))  # type: ignore[return-value]


@app.post("/v1/inspect-next", response_model=InspectNextResponse)
async def inspect_next(payload: InspectNextRequest) -> InspectNextResponse:
    try:
        return inspect_next_logic(payload)
    except NodeLimitExceeded as exc:
        # 树构建、合取观测约束或构建候选分支时累计非终端节点超过限制
        return _bad_request(str(exc))  # type: ignore[return-value]
    except ValueError as exc:
        # 未知引用、环、非法观测键/候选、观测矛盾或联合概率为 0 等语义错误
        return _bad_request(str(exc))  # type: ignore[return-value]


@app.post("/v1/likely-cause", response_model=LikelyCauseResponse)
async def likely_cause(payload: LikelyCauseRequest) -> LikelyCauseResponse:
    try:
        return likely_cause_logic(payload)
    except NodeLimitExceeded as exc:
        # 树构建或合取观测约束时累计非终端节点超过限制
        return _bad_request(str(exc))  # type: ignore[return-value]
    except ValueError as exc:
        # 未知引用、环、非法观测键、条件矛盾或联合概率为 0 等语义错误
        return _bad_request(str(exc))  # type: ignore[return-value]
