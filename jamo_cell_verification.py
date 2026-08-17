# -*- coding: utf-8 -*-
SB,VC,TC=0xAC00,21,28; NC=VC*TC
L=list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
V=list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
T=[""]+list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
COMPLEX_V=set("ㅘㅙㅚㅝㅞㅟㅢ"); DIPH_V=set("ㅐㅒㅔㅖ")
CLUSTER_T=set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")
# 리뷰 C-2 반영: 음운 분류가 아니라 '시각 복잡도' 기반 초성군
SIMPLE_ONSET=set("ㄱㄴㄷㄹㅁㅂㅅㅇㅈ")      # 기본형 9종
ASPIR_ONSET =set("ㅊㅋㅌㅍㅎ")              # 가획 5종
TENSE_ONSET =set("ㄲㄸㅃㅆㅉ")              # 쌍자음 5종

def vc(v): return "complex_V" if v in COMPLEX_V else ("diph_V" if v in DIPH_V else "simple_V")
def tc(t): return "no_T" if t=="" else ("cluster_T" if t in CLUSTER_T else "simple_T")
def oc(l): return "simple_O" if l in SIMPLE_ONSET else ("aspir_O" if l in ASPIR_ONSET else "tense_O")

from collections import Counter
c27=Counter(); c18=Counter(); c9=Counter()
for i in range(11172):
    s=i; l,v,t = L[s//NC], V[(s%NC)//TC], T[s%TC]
    a,b,d = vc(v), tc(t), oc(l)
    c27[(a,b,d)]+=1
    c18[("simple_V" if a in("simple_V","diph_V") else a, b, d)]+=1
    c9[(a,b)]+=1

print("=== A-1 검증: 27셀(3모음×3종성×3초성군) 셀 크기 ===")
print(f"{'셀':<34}{'후보':>7}  {'40개 확보?':<10}")
short=[]
for a in ["simple_V","diph_V","complex_V"]:
    for b in ["no_T","simple_T","cluster_T"]:
        for d in ["simple_O","aspir_O","tense_O"]:
            n=c27[(a,b,d)]
            ok = "OK" if n>=40 else ("부족" if n>=25 else "심각")
            if n<40: short.append(((a,b,d),n))
            if b=="no_T" or n<60:
                print(f"{a+' × '+b+' × '+d:<34}{n:>7}  {ok:<10}")
print(f"\n27셀 중 40개 미달 셀: {len(short)}개 / 27")
print(f"최소 셀: {min(c27.values())}개")

print("\n=== 18셀(이중자모를 단순모음에 병합) ===")
short18=[k for k,n in c18.items() if n<40]
print(f"셀 수 18, 최소 셀 {min(c18.values())}개, 40개 미달 {len(short18)}개")
for k,n in sorted(c18.items(), key=lambda x:x[1])[:4]:
    print(f"  {k}: {n}")

print("\n=== 9셀(초성군 제거) ===")
print(f"최소 셀 {min(c9.values())}개, 40개 미달 {len([n for n in c9.values() if n<40])}개")

print("\n=== A-2 검증: 균형 세트에서의 종성 우연 일치 ===")
print("자연 분포(종성없음 55%)  : 우연일치 = .55^2 + .45^2/27 = %.1f%%" % ((0.55**2+0.45**2/27)*100))
print("균형 세트(각 유형 1/3)   : 우연일치 = 3 × (1/3)^2 (유형 내 세부는 별도)")
p_bal = (1/3)**2 + (1/3)**2/27 + (1/3)**2/11   # 무종성 정확일치 / 단순27종 / 겹11종
print("  ≈ %.1f%%  (자연분포 31.0%%의 약 %.1f분의 1)" % (p_bal*100, 0.310/p_bal))
print("\n'받침 전부 탈락' 전략의 종성 정확도:")
print("  자연 분포 세트: 55%%")
print("  균형 세트     : %.0f%%  ← 설계 자체가 방어선" % (100/3))
