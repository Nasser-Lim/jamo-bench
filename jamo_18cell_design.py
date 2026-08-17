# -*- coding: utf-8 -*-
"""v5 최종 셀 설계 검증"""
SB,VC,TC=0xAC00,21,28; NC=VC*TC
L=list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
V=list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
T=[""]+list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
CX=set("ㅘㅙㅚㅝㅞㅟㅢ"); CL=set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")
SIMPLE_O=set("ㄱㄴㄷㄹㅁㅂㅅㅇㅈ"); ASPIR_O=set("ㅊㅋㅌㅍㅎ"); TENSE_O=set("ㄲㄸㅃㅆㅉ")
from collections import Counter
c=Counter()
for i in range(11172):
    l,v,t=L[i//NC],V[(i%NC)//TC],T[i%TC]
    a="complex_V" if v in CX else "simple_V"          # 이중자모 병합
    b="no_T" if t=="" else ("cluster_T" if t in CL else "simple_T")
    d="simple_O" if l in SIMPLE_O else ("aspir_O" if l in ASPIR_O else "tense_O")
    c[(a,b,d)]+=1
print("=== v5 확정 설계: 2모음 × 3종성 × 3초성군 = 18셀 ===")
print(f"{'모음':<11}{'종성':<11}{'초성군':<11}{'후보':>6}{'  n=30?':<8}")
for a in ["simple_V","complex_V"]:
    for b in ["no_T","simple_T","cluster_T"]:
        for d in ["simple_O","aspir_O","tense_O"]:
            n=c[(a,b,d)]
            print(f"{a:<11}{b:<11}{d:<11}{n:>6}{'  OK' if n>=30 else '  부족':<8}")
mn=min(c.values())
print(f"\n최소 셀 {mn}개, 목표 unique 30 → {'전 셀 충족' if mn>=30 else '미달'}")
print(f"총 생성량: 18셀 × 30 unique × 3 samples × 2 template = {18*30*3*2:,}장")
