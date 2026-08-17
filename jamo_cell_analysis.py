# -*- coding: utf-8 -*-
SBase, LCount, VCount, TCount = 0xAC00, 19, 21, 28
NCount = VCount * TCount

L = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
V = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
T = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

# 시각적 복잡도 분류
COMPLEX_V = set("ㅘㅙㅚㅝㅞㅟㅢ")       # 3획 이상 결합 모음 (가로+세로 합성)
DIPH_V     = set("ㅐㅒㅔㅖ")            # 이중자모지만 시각적으론 단순 세로형
CLUSTER_T  = set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")  # 겹받침 11종

def decomp(ch):
    s = ord(ch) - SBase
    return L[s//NCount], V[(s%NCount)//TCount], T[s%TCount]

def vclass(v):
    if v in COMPLEX_V: return "complex_V"
    if v in DIPH_V:    return "diph_V"
    return "simple_V"

def tclass(t):
    if t == "": return "no_T"
    if t in CLUSTER_T: return "cluster_T"
    return "simple_T"

from collections import Counter
cells = Counter()
for i in range(11172):
    ch = chr(SBase + i)
    l, v, t = decomp(ch)
    cells[(vclass(v), tclass(t))] += 1

print("=== 전체 11,172 음절의 (모음유형 × 종성유형) 셀 크기 ===")
vs = ["simple_V","diph_V","complex_V"]
ts = ["no_T","simple_T","cluster_T"]
print(f"{'':12}" + "".join(f"{t:>12}" for t in ts) + f"{'합계':>10}")
for v in vs:
    row = [cells[(v,t)] for t in ts]
    print(f"{v:12}" + "".join(f"{n:>12,}" for n in row) + f"{sum(row):>10,}")
print(f"{'합계':12}" + "".join(f"{sum(cells[(v,t)] for v in vs):>12,}" for t in ts) + f"{11172:>10,}")

print()
print("겹받침 음절 총수:", sum(cells[(v,'cluster_T')] for v in vs), f"({sum(cells[(v,'cluster_T')] for v in vs)/11172*100:.1f}%)")
print("복합모음 음절 총수:", sum(cells[('complex_V',t)] for t in ts), f"({sum(cells[('complex_V',t)] for t in ts)/11172*100:.1f}%)")
print("복합모음+겹받침 교차:", cells[('complex_V','cluster_T')])
