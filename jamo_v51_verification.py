# -*- coding: utf-8 -*-
SB,VC,TC=0xAC00,21,28; NC=VC*TC
L=list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
V=list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
T=[""]+list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
CX=set("ㅘㅙㅚㅝㅞㅟㅢ"); DERIV=set("ㅐㅒㅔㅖ")
CLUSTER=set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ"); TENSED=set("ㄲㅆ")
from collections import Counter
c4=Counter()
for i in range(11172):
    v,t=V[(i%NC)//TC],T[i%TC]
    a="complex_V" if v in CX else "simple_V"
    if t=="": b="none"
    elif t in TENSED: b="tensed_double"
    elif t in CLUSTER: b="cluster_mixed"
    else: b="simple_single"
    c4[(a,b)]+=1
print("=== 종성 4분류 검증 (리뷰 #4) ===")
for a in ["simple_V","complex_V"]:
    for b in ["none","simple_single","tensed_double","cluster_mixed"]:
        print(f"  {a:<11}{b:<15}{c4[(a,b)]:>6}")
print(f"\n쌍받침(ㄲㅆ) 총 음절: {sum(c4[(a,'tensed_double')] for a in ['simple_V','complex_V'])}")
print("→ 현재 simple_single에 섞여 있어 겹받침 gap을 희석시킴. 별도 contrast(H2b) 필요")

print("\n=== 이중자모(ㅐㅒㅔㅖ) 규모 (리뷰 #5) ===")
n_deriv=sum(1 for i in range(11172) if V[(i%NC)//TC] in DERIV)
print(f"  vertical_derived 음절: {n_deriv} ({n_deriv/11172:.1%})")
print("→ ㅐ↔ㅔ는 한글 최대 혼동쌍. 병합하되 메타데이터 축 유지 필요")

print("\n=== v5.1 예산 재계산 (Cross ⊂ Core) ===")
core = 18*30*3*2
cross_other = 4*250*2          # Seedream은 Core에서 subset 추출, 재생성 없음
cross_struct = 2*6*30*2
word = 250*2
t3 = 6*20
buf = 2000
tot = core+cross_other+cross_struct+word+t3+buf
print(f"  Core (Seedream)        {core:>6,}")
print(f"  Cross 타 4모델          {cross_other:>6,}   (Seedream 중복 생성 제거로 500장 절약)")
print(f"  Cross-Struct 2모델      {cross_struct:>6,}")
print(f"  Word                   {word:>6,}")
print(f"  T3 exploratory          {t3:>6,}")
print(f"  버퍼                    {buf:>6,}")
print(f"  합계                   {tot:>6,}  / 예산 약 16,000장")
print(f"  여유 {16000-tot:,}장 → Core n_samples 3→4 가능 ({18*30*4*2-core:,}장 추가)")
