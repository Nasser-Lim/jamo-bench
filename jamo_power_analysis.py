import math
# 이항 비율 비교 검정력 (two-proportion z-test), alpha=0.05, power=0.80
def n_per_group(p1, p2, alpha=0.05, power=0.80):
    from statistics import NormalDist
    za = NormalDist().inv_cdf(1-alpha/2); zb = NormalDist().inv_cdf(power)
    pbar = (p1+p2)/2
    num = (za*math.sqrt(2*pbar*(1-pbar)) + zb*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2
    return math.ceil(num/(p1-p2)**2)

print("=== 셀당 최소 표본 수 (alpha=.05, power=.80) ===")
print(f"{'비교 시나리오':<34}{'p1':>7}{'p2':>7}{'셀당 n':>9}")
for label,p1,p2 in [
    ("종성 유/무 (큰 효과)",      0.80, 0.55),
    ("종성 유/무 (중간 효과)",    0.70, 0.55),
    ("겹받침 vs 단순종성 (큰)",   0.55, 0.20),
    ("겹받침 vs 단순종성 (중간)", 0.45, 0.30),
    ("복합모음 vs 단순 (중간)",   0.60, 0.45),
    ("복합모음 vs 단순 (작은)",   0.55, 0.47),
    ("모델 A vs B (10%p)",        0.55, 0.45),
]:
    print(f"{label:<34}{p1:>7.2f}{p2:>7.2f}{n_per_group(p1,p2):>9,}")

print()
print("=== 요인설계 총 표본 (셀당 n=120 가정) ===")
print("Plan A (4 Tier × 3 모음 × 3 종성 = 36셀):", 36*120, "장")
print("Plan B (4 Tier + 3 모음 + 3 종성, 가법):  약", 12*120, "장 (주효과만)")
print("Plan C (3 모음 × 3 종성 = 9셀, Tier 제거):", 9*120, "장")
print("Plan C-확장 (9셀 × 초성군 3 = 27셀):", 27*120, "장")
