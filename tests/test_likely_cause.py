"""POST /v1/likely-cause：最可能诱因（最大先验乘积赋值）的语义、
精确性、并列裁决与 422 校验。

用穷举真值表（枚举全部事件状态）作为参考实现做交叉验证。
"""

import itertools
import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------- 穷举真值表参考实现 ----------

def brute_likely(ev_specs, gate_specs, top, observations):
    """枚举全部事件赋值，返回 (P(C), P(A*)/P(C), {事件 id: 最优取值})。

    条件 C = top 成立且全部读数成立；A* 取独立先验乘积最大的完整
    赋值，等权时按事件 id 字典序（false 优先）。P(C)=0 时返回 None。
    """
    ev_ids = sorted(ev_specs)
    probs = {e: Fraction(ev_specs[e]) for e in ev_ids}
    gates = {g[0]: (g[1], g[2]) for g in gate_specs}

    def evaluate(state, node):
        if node in state:
            return state[node]
        op, inputs = gates[node]
        vals = [evaluate(state, i) for i in inputs]
        if op == "AND":
            return all(vals)
        if op == "OR":
            return any(vals)
        return not vals[0]

    p_c = Fraction(0)
    best = None  # (最大权重, 0/1 序列；0 < 1 即 false 优先)
    for bits in itertools.product((0, 1), repeat=len(ev_ids)):
        state = dict(zip(ev_ids, (bool(b) for b in bits)))
        if not evaluate(state, top):
            continue
        if any(
            evaluate(state, oid) != value
            for oid, value in observations.items()
        ):
            continue
        weight = Fraction(1)
        for e in ev_ids:
            weight *= probs[e] if state[e] else (Fraction(1) - probs[e])
        p_c += weight
        if (
            best is None
            or weight > best[0]
            or (weight == best[0] and bits < best[1])
        ):
            best = (weight, bits)

    if p_c == 0:
        return None
    assert best is not None
    return (
        p_c,
        best[0] / p_c,
        dict(zip(ev_ids, (bool(b) for b in best[1]))),
    )


def f2s(f):
    return f"{f.numerator}/{f.denominator}"


def make_body(ev_specs, gate_specs, top, observations=None):
    body = {
        "events": [{"id": k, "p": v} for k, v in ev_specs.items()],
        "gates": [
            {"id": i, "op": op, "inputs": ins}
            for i, op, ins in gate_specs
        ],
        "top": top,
    }
    if observations is not None:
        body["observations"] = observations
    return body


# ---------- 基本语义：README 示例 ----------

MOTOR_EV = {
    "motor_overheat": "1/100",
    "coolant_loss": "2/100",
    "pump_fail": "1/200",
}
MOTOR_GATES = [
    ("cooling_fail", "OR", ["coolant_loss", "pump_fail"]),
    ("top", "OR", ["motor_overheat", "cooling_fail"]),
]


def test_readme_example():
    # 观测电机未过热且顶事件发生：最可能诱因是冷却液泄漏
    body = make_body(
        MOTOR_EV, MOTOR_GATES, "top",
        {"motor_overheat": False, "top": True},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    exp = brute_likely(
        MOTOR_EV, MOTOR_GATES, "top",
        {"motor_overheat": False, "top": True},
    )
    assert exp is not None
    p_c, posterior, assignment = exp
    data = r.json()
    assert data["condition_probability"] == f2s(p_c)
    assert data["condition_probability"] == "24651/1000000"
    assert data["assignment_posterior"] == f2s(posterior)
    assert data["assignment_posterior"] == "199/249"
    assert data["assignment"] == assignment
    assert data["assignment"] == {
        "coolant_loss": True,
        "motor_overheat": False,
        "pump_fail": False,
    }
    # assignment 必须按事件 id 字典序排列
    assert list(data["assignment"]) == sorted(MOTOR_EV)


def test_no_observations_picks_best_top_state():
    # 无读数：C 即 top 本身；OR 顶下三个“单故障”状态取先验最大者
    body = make_body(MOTOR_EV, MOTOR_GATES, "top", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    exp = brute_likely(MOTOR_EV, MOTOR_GATES, "top", {})
    assert exp is not None
    p_c, posterior, assignment = exp
    assert data["condition_probability"] == f2s(p_c) == "34651/1000000"
    assert data["assignment_posterior"] == f2s(posterior)
    assert data["assignment_posterior"] == "19701/34651"
    assert data["assignment"] == assignment
    assert data["assignment"] == {
        "coolant_loss": True,
        "motor_overheat": False,
        "pump_fail": False,
    }


def test_observations_may_be_omitted():
    body = make_body(MOTOR_EV, MOTOR_GATES, "top")
    assert "observations" not in body
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["condition_probability"] == "34651/1000000"


def test_single_event_top():
    # top 就是事件 a，p<1/2：条件要求 a=true，仍须取 true
    body = make_body({"a": "3/7"}, [], "a", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data == {
        "assignment": {"a": True},
        "condition_probability": "3/7",
        "assignment_posterior": "1/1",
    }


def test_not_gate_participates():
    # top = NOT a：最可能状态为 a=false
    body = make_body(
        {"a": "1/3"}, [("g", "NOT", ["a"])], "g", {}
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["assignment"] == {"a": False}
    assert data["condition_probability"] == "2/3"
    assert data["assignment_posterior"] == "1/1"


def test_not_gate_observed():
    # 观测 NOT a 为 true 等价于 a=false
    body = make_body(
        {"a": "3/7"}, [("na", "NOT", ["a"])], "na", {"na": True}
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["assignment"] == {"a": False}
    assert data["condition_probability"] == "4/7"
    assert data["assignment_posterior"] == "1/1"


def test_unconstrained_free_events_take_likely_side():
    # 顶事件 a 与 b 无关：b 自由，p=9/10 取 true；a 被条件要求 true
    body = make_body(
        {"a": "1/2", "b": "9/10"}, [], "a", {}
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["assignment"] == {"a": True, "b": True}
    assert data["condition_probability"] == "1/2"
    assert data["assignment_posterior"] == "9/10"


def test_free_event_at_half_takes_false_first():
    # 自由变量 p=1/2 因子并列时取 false
    body = make_body(
        {"a": "1/2", "b": "1/2"},
        [("g", "OR", ["a", "a"])], "g", {},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    # a 被 OR 要求 true；b 自由、1/2 并列 -> false
    assert data["assignment"] == {"a": True, "b": False}
    assert data["condition_probability"] == "1/2"
    assert data["assignment_posterior"] == "1/2"


def test_xor_tie_breaks_false_first_lexicographically():
    # top = a XOR b（非平凡），两事件 p=1/2：
    # (a,b)=(0,1) 与 (1,0) 等权 1/4，字典序 false 优先 -> (0,1)
    body = make_body(
        {"a": "1/2", "b": "1/2"},
        [("o", "OR", ["a", "b"]),
         ("ad", "AND", ["a", "b"]),
         ("n", "NOT", ["ad"]),
         ("top", "AND", ["o", "n"])],
        "top", {},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["assignment"] == {"a": False, "b": True}
    assert data["condition_probability"] == "1/2"
    # P(A*)/P(C) = (1/4)/(1/2)
    assert data["assignment_posterior"] == "1/2"


def test_tie_break_uses_ascii_id_order_not_numeric():
    # 事件 id 的字典序按 ASCII 比较：e0 < e1 < e10 < e2；
    # 11 个变量的奇校验中并列的单 1 状态取“最后”变量，即排序后的 e9。
    n = 11
    ev_specs = {f"e{i}": "1/2" for i in range(n)}
    gates = []
    prev = "e0"
    for i in range(1, n):
        o, a, nn, x = f"o{i}", f"a{i}", f"n{i}", f"x{i}"
        gates += [
            (o, "OR", [prev, f"e{i}"]),
            (a, "AND", [prev, f"e{i}"]),
            (nn, "NOT", [a]),
            (x, "AND", [o, nn]),
        ]
        prev = x
    body = make_body(ev_specs, gates, prev, {prev: True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    true_ids = [k for k, v in data["assignment"].items() if v]
    assert true_ids == ["e9"]  # sorted(...) 的最后一个元素
    assert sorted(ev_specs)[-1] == "e9"
    # 权重为 2^-11，条件概率 1/2，后验 2^-10
    assert data["condition_probability"] == "1/2"
    assert data["assignment_posterior"] == f"1/{2 ** (n - 1)}"


def test_shared_reference_shares_assignment():
    # (a AND b) OR (a AND c)：观测 g1=true、g2=false 推出 a=b=1,c=0
    ev = {"a": "1/2", "b": "1/3", "c": "1/5"}
    gates = [
        ("g1", "AND", ["a", "b"]),
        ("g2", "AND", ["a", "c"]),
        ("top", "OR", ["g1", "g2"]),
    ]
    body = make_body(ev, gates, "top", {"g1": True, "g2": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    exp = brute_likely(ev, gates, "top", {"g1": True, "g2": False})
    assert exp is not None
    assert data["assignment"] == exp[2] == {
        "a": True, "b": True, "c": False,
    }
    assert data["condition_probability"] == f2s(exp[0])
    assert data["assignment_posterior"] == "1/1"


def test_constant_true_condition():
    # top = a OR NOT a 恒真：C 恒成立，自由变量各取先验较大侧
    body = make_body(
        {"a": "1/2", "b": "3/4"},
        [("na", "NOT", ["a"]),
         ("top", "OR", ["a", "na"])],
        "top", {},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["assignment"] == {"a": False, "b": True}
    assert data["condition_probability"] == "1/1"
    assert data["assignment_posterior"] == "3/8"


def test_fractions_are_reduced():
    body = make_body(
        {"a": "2/4", "b": "4/8"},
        [("g", "AND", ["a", "b"])], "g", {"g": True},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["condition_probability"] == "1/4"
    assert data["assignment_posterior"] == "1/1"


def test_observed_unreachable_gate_still_built():
    # top 不经过 g，但观测 g=true：条件须同时满足 a=1 与 a∧b=1
    body = make_body(
        {"a": "1/2", "b": "1/3"},
        [("g", "AND", ["a", "b"])], "a",
        {"g": True},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    exp = brute_likely(
        {"a": "1/2", "b": "1/3"},
        [("g", "AND", ["a", "b"])], "a",
        {"g": True},
    )
    assert exp is not None
    assert data["condition_probability"] == f2s(exp[0]) == "1/6"
    assert data["assignment"] == {"a": True, "b": True}
    assert data["assignment_posterior"] == "1/1"


# ---------- 随机电路穷举对照 ----------

def test_random_circuits_against_truth_table():
    rng = random.Random(20260929)
    checked = 0
    while checked < 40:
        n = rng.randint(1, 10)
        ev_ids = [f"e{i}" for i in range(n)]
        ev_specs = {}
        for e in ev_ids:
            d = rng.randint(1, 8)
            ev_specs[e] = f"{rng.randint(0, d)}/{d}"
        gates = []
        available = list(ev_ids)
        for j in range(rng.randint(0, min(16, 3 * n))):
            gid = f"g{j}"
            if len(available) < 2:
                op = "NOT"
            else:
                op = rng.choice(["AND", "OR", "NOT"])
            if op == "NOT":
                inputs = [rng.choice(available)]
            else:
                k = rng.randint(2, min(8, len(available)))
                inputs = rng.sample(available, k)
            gates.append((gid, op, inputs))
            available.append(gid)
        top = rng.choice(available)

        # 0~16 个观测，键可取自事件或门
        candidates = list(ev_ids) + [g[0] for g in gates]
        k = min(len(candidates), rng.randint(0, 16))
        chosen = rng.sample(candidates, k)
        observations = {oid: rng.choice([True, False]) for oid in chosen}

        body = make_body(ev_specs, gates, top, observations)
        r = client.post("/v1/likely-cause", json=body)
        exp = brute_likely(ev_specs, gates, top, observations)
        if exp is None:
            assert r.status_code == 422, (
                r.text, ev_specs, gates, observations
            )
            continue
        assert r.status_code == 200, (
            r.text, ev_specs, gates, observations
        )
        p_c, posterior, assignment = exp
        data = r.json()
        assert data["condition_probability"] == f2s(p_c)
        assert data["assignment_posterior"] == f2s(posterior)
        assert data["assignment"] == assignment
        assert list(data["assignment"]) == sorted(ev_specs)
        # 后验必须为合法概率
        value = Fraction(data["assignment_posterior"])
        assert 0 <= value <= 1
        checked += 1


# ---------- P(C)=0 ----------

def test_structural_contradiction_returns_422():
    # 顶事件要求 a=1，读数却要求 a=0：结构上永不可同真
    body = make_body({"a": "1/2"}, [], "a", {"a": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "condition_probability=0" in detail and "结构" in detail


def test_structural_contradiction_gate_readings_returns_422():
    # a=true 与 NOT a=true 永不同时成立
    body = make_body(
        {"a": "1/2"},
        [("na", "NOT", ["a"])], "na",
        {"a": True, "na": True},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "condition_probability=0" in detail and "矛盾" in detail


def test_zero_prior_condition_returns_422():
    # 结构可满足（a=1），但 P(a)=0，先验下无正概率赋值
    body = make_body({"a": "0/1"}, [], "a", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "condition_probability=0" in detail and "先验" in detail


def test_one_prior_event_required_false_returns_422():
    # 顶事件为 NOT a，a 必发生（p=1），故 top 恒假
    body = make_body(
        {"a": "1/1"}, [("g", "NOT", ["a"])], "g", {}
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "condition_probability=0" in r.json()["detail"]


def test_impossible_observation_by_prior_returns_422():
    # 读数要求 p=1 的事件为 false：结构可满足但先验概率为 0
    body = make_body({"a": "1/1", "b": "1/2"}, [], "a", {"a": False})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "condition_probability=0" in r.json()["detail"]


# ---------- 422 校验路径 ----------

def test_unknown_observation_key():
    body = make_body({"a": "1/2"}, [], "a", {"nope": True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "observations" in detail and "nope" in detail


@pytest.mark.parametrize("bad_value", [1, 0, "true", None, 1.0])
def test_non_boolean_observation_value_rejected(bad_value):
    body = make_body({"a": "1/2"}, [], "a", {"a": bad_value})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422


def test_empty_observations_allowed():
    body = make_body({"a": "1/2"}, [], "a", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text


def test_too_many_observations_rejected():
    ev_specs = {f"e{i}": "1/2" for i in range(17)}
    body = make_body(
        ev_specs, [], "e0",
        {f"e{i}": True for i in range(17)},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422


def test_sixteen_observations_allowed():
    ev_specs = {f"e{i}": "1/2" for i in range(16)}
    body = make_body(
        ev_specs, [], "e0",
        {f"e{i}": True for i in range(16)},
    )
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 200, r.text


def test_extra_field_rejected():
    body = make_body({"a": "1/2"}, [], "a", {})
    body["extra"] = 1
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422


def test_original_validation_still_applies():
    # 未知 top
    body = make_body({"a": "1/2"}, [], "nope", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "顶层" in r.json()["detail"]
    # 环
    body = make_body(
        {"a": "1/2"},
        [("g", "OR", ["a", "g"])], "g", {"g": True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "环" in r.json()["detail"]
    # 非法概率
    body = make_body({"a": "3/2"}, [], "a", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "概率" in r.json()["detail"]
    # 非 ASCII id
    body = make_body({"事件": "1/2"}, [], "事件", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    # 非法门元数
    body = make_body(
        {"a": "1/2"}, [("g", "NOT", ["a", "a"])], "g", {})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422 and "NOT" in r.json()["detail"]


def test_node_limit_in_condition_conjunction(monkeypatch):
    # 树本身节点不多，但合取条件约束（NOT + AND 的 ITE 展开）可能
    # 新增节点；调小上限后应返回 422
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 5)
    evs = {f"e{i}": "1/2" for i in range(8)}
    gates = []
    prev = "e0"
    for i in range(1, 8):
        o, a, n, x = f"o{i}", f"a{i}", f"n{i}", f"x{i}"
        gates.append((o, "OR", [prev, f"e{i}"]))
        gates.append((a, "AND", [prev, f"e{i}"]))
        gates.append((n, "NOT", [a]))
        gates.append((x, "AND", [o, n]))
        prev = x
    body = make_body(evs, gates, prev, {prev: True})
    r = client.post("/v1/likely-cause", json=body)
    assert r.status_code == 422
    assert "非终端节点" in r.json()["detail"]
