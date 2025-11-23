from dataclasses import dataclass
import math
import os
import json
from datetime import date

from flask import Flask, render_template, request

app = Flask(__name__)

# =========================
# アクセスカウンター用
# =========================

COUNTER_FILE = "counter.json"


def load_counter():
    """counter.json を読み込む。なければデフォルト値を返す。"""
    today_str = date.today().isoformat()
    if not os.path.exists(COUNTER_FILE):
        return {"date": today_str, "today": 0, "total": 0}

    try:
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # 壊れてた場合はリセット
        return {"date": today_str, "today": 0, "total": 0}

    # 必要なキーが無ければ補完
    if "date" not in data or "today" not in data or "total" not in data:
        data = {"date": today_str, "today": 0, "total": 0}

    return data


def save_counter(data: dict):
    """counter.json に書き込む"""
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_counter():
    """
    今日の日付を見て、本日・累計カウントを更新して返す
    戻り値: (today_count, total_count)
    """
    today_str = date.today().isoformat()
    data = load_counter()

    if data["date"] != today_str:
        # 日付が変わっていたら、本日カウントをリセット
        data["date"] = today_str
        data["today"] = 0

    data["today"] += 1
    data["total"] += 1

    save_counter(data)
    return data["today"], data["total"]


# =========================
# データ定義・計算ロジック
# =========================

# 楽曲の難易度ごとの基本ポイント
# （ペンライト無・トロフィー無・スコアD・コンボD）
BASE_POINTS = {
    "EASY": 120,
    "NORMAL": 150,
    "HARD": 190,
    "EXPERT": 230,
}

# ランクの昇順（内部用） D が一番下
RANK_ORDER_BASE = ["D", "C", "B", "A", "S", "SS"]
# 表示用（SS → D）
RANK_DISPLAY = list(reversed(RANK_ORDER_BASE))

# ペンライト
PENLIGHT_OPTIONS = ["有り", "無し"]

# トロフィーとボーナス率
TROPHY_OPTIONS = ["金", "銀", "銅", "緑", "青", "無"]
TROPHY_RATE = {
    "金": 1.0,
    "銀": 0.7,
    "銅": 0.4,
    "緑": 0.2,
    "青": 0.1,
    "無": 0.0,
}

# 消費倍率
CONSUME_LIST = [1, 3, 5]


@dataclass
class Track:
    """1パターン分の楽曲ポイント情報"""
    point: int          # 最終ポイント
    difficulty: str     # 難易度 EASY/NORMAL/HARD/EXPERT
    score_rank: str     # スコアランク SS〜D
    combo_rank: str     # コンボランク SS〜D
    consume: int        # 消費倍率 1/3/5


def rank_step(rank: str) -> int:
    """D=0, C=1, B=2, A=3, S=4, SS=5 を返す"""
    return RANK_ORDER_BASE.index(rank)


def calc_point(base: int,
               score_rank: str,
               combo_rank: str,
               penlight: str,
               trophy: str,
               consume: int) -> int:
    """
    最終ポイント ＝
      ceil(基礎×消費)
    ＋ ceil(基礎×スコア補正×消費)
    ＋ ceil(基礎×コンボ補正×消費)
    ＋ ceil(基礎×ペンライト補正×消費)
    ＋ ceil(基礎×トロフィー補正×消費)
    """

    # ランク補正（1段 5％）
    score_rate = 0.05 * rank_step(score_rank)
    combo_rate = 0.05 * rank_step(combo_rank)

    # ペンライト
    pen_rate = 0.5 if penlight == "有り" else 0.0

    # トロフィー
    tro_rate = TROPHY_RATE[trophy]

    # それぞれの項目で倍率を掛けてから切り上げ
    base_term = math.ceil(base * consume)
    score_term = math.ceil(base * score_rate * consume)
    combo_term = math.ceil(base * combo_rate * consume)
    pen_term = math.ceil(base * pen_rate * consume)
    tro_term = math.ceil(base * tro_rate * consume)

    return base_term + score_term + combo_term + pen_term + tro_term


def generate_points(penlight: str, trophy: str):
    """
    ペンライト／トロフィー条件から、
    ・432個の Track リスト（生データ）
    を作る
    """
    tracks: list[Track] = []

    for consume in CONSUME_LIST:
        for diff, base in BASE_POINTS.items():
            for score_rank in RANK_DISPLAY:        # 表は SS→D
                for combo_rank in RANK_DISPLAY:    # 表は SS→D
                    p = calc_point(
                        base=base,
                        score_rank=score_rank,
                        combo_rank=combo_rank,
                        penlight=penlight,
                        trophy=trophy,
                        consume=consume,
                    )

                    tracks.append(
                        Track(
                            point=p,
                            difficulty=diff,
                            score_rank=score_rank,
                            combo_rank=combo_rank,
                            consume=consume,
                        )
                    )

    return tracks


def build_unique_tracks(penlight: str, trophy: str):
    """
    ・432件の生Track一覧
    ・(point, consume, diff)ごとに1枠だけの unique_tracks
    ・同じ枠の中の全ランクパターン variant_map
    をまとめて返す
    """
    tracks_full = generate_points(penlight, trophy)

    # 同じポイント＋消費＋難易度ごとのバリエーション一覧を作る
    variant_map: dict[tuple, list[Track]] = {}
    for tr in tracks_full:
        key = (tr.point, tr.consume, tr.difficulty)
        variant_map.setdefault(key, []).append(tr)

    # 探索用：ユニークな枠だけに圧縮
    unique_tracks = []
    for key, variants in variant_map.items():
        unique_tracks.append(variants[0])  # 代表1件

    # 見やすさのためポイント昇順にソート
    unique_tracks.sort(key=lambda t: (t.point, t.consume, t.difficulty))

    return tracks_full, unique_tracks, variant_map


def search_solutions(unique_tracks: list[Track],
                     variant_map: dict[tuple, list[Track]],
                     target: int,
                     max_results: int = 20):
    """
    1〜4枠までで target に一致する組み合わせを探す。
    unique_tracks を元にインデックスの組み合わせを探し、
    テンプレートで使いやすい dict 形式のリストにして返す。
    """
    tracks = unique_tracks
    n = len(tracks)
    points = [t.point for t in tracks]
    results: list[list[int]] = []
    seen_solutions = set()

    def add_solution(idxs: list[int]) -> bool:
        ord_idxs = tuple(sorted(idxs))
        if ord_idxs in seen_solutions:
            return True
        seen_solutions.add(ord_idxs)
        results.append(list(ord_idxs))
        return len(results) < max_results

    # --- 1枠 ---
    for i in range(n):
        if points[i] == target:
            if not add_solution([i]):
                break

    # --- 2枠 ---
    if len(results) < max_results:
        for i in range(n):
            pi = points[i]
            for j in range(i + 1, n):
                if pi + points[j] == target:
                    if not add_solution([i, j]):
                        break
            if len(results) >= max_results:
                break

    # --- 3枠 ---
    if len(results) < max_results:
        for i in range(n):
            pi = points[i]
            for j in range(i + 1, n):
                pj = points[j]
                remain = target - pi - pj
                for k in range(j + 1, n):
                    if points[k] == remain:
                        if not add_solution([i, j, k]):
                            break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

    # --- 4枠（ペア和） ---
    if len(results) < max_results:
        pair_sums: dict[int, list[tuple[int, int]]] = {}
        for i in range(n):
            for j in range(i + 1, n):
                s = points[i] + points[j]
                pair_sums.setdefault(s, []).append((i, j))

        for s1, pairs1 in pair_sums.items():
            s2 = target - s1
            if s2 not in pair_sums:
                continue
            pairs2 = pair_sums[s2]
            for i, j in pairs1:
                for k, l in pairs2:
                    idxs = [i, j, k, l]
                    if len(set(idxs)) < 4:
                        continue
                    if not add_solution(idxs):
                        break
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

    # --- テンプレート用に整形 ---
    solution_dicts = []
    for ans_no, idxs in enumerate(results, start=1):
        total = sum(points[i] for i in idxs)
        tracks_for_solution = []
        for i in idxs:
            tr = tracks[i]
            key = (tr.point, tr.consume, tr.difficulty)
            variants = variant_map.get(key, [tr])

            variant_list = [
                {
                    "score_rank": v.score_rank,
                    "combo_rank": v.combo_rank,
                }
                for v in variants
            ]

            tracks_for_solution.append(
                {
                    "point": tr.point,
                    "consume": tr.consume,
                    "difficulty": tr.difficulty,
                    "variants": variant_list,
                }
            )

        solution_dicts.append(
            {
                "no": ans_no,
                "total_points": total,
                "count": len(idxs),
                "tracks": tracks_for_solution,
            }
        )

    return solution_dicts


# =========================
# Flask ルーティング
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    error = None
    solutions = []

    # アクセスカウンタ更新
    try:
        today_count, total_count = update_counter()
    except Exception:
        today_count, total_count = None, None

    # デフォルトの条件
    penlight = PENLIGHT_OPTIONS[0]
    trophy = TROPHY_OPTIONS[0]
    target_str = ""

    if request.method == "POST":
        penlight = request.form.get("penlight", penlight)
        trophy = request.form.get("trophy", trophy)
        target_str = request.form.get("target", "").strip()

        # 入力チェック
        if not target_str:
            error = "目的の合計を入力してください。"
        else:
            try:
                target = int(target_str)
                if target <= 0 or target > 20000:
                    error = "目的の合計は 1〜20000 の範囲で入力してください。"
            except ValueError:
                error = "目的の合計は整数で入力してください。"

        if not error:
            # 計算
            _, unique_tracks, variant_map = build_unique_tracks(penlight, trophy)
            solutions = search_solutions(unique_tracks, variant_map, target)

            if not solutions:
                message = "1〜4枠の範囲では一致する組み合わせは見つかりませんでした。"
            else:
                message = f"{len(solutions)}個の解が見つかりました。（上限20）"

    return render_template(
        "index.html",
        penlight_options=PENLIGHT_OPTIONS,
        trophy_options=TROPHY_OPTIONS,
        penlight_selected=penlight,
        trophy_selected=trophy,
        target_value=target_str,
        message=message,
        error=error,
        solutions=solutions,
        today_count=today_count,
        total_count=total_count,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

