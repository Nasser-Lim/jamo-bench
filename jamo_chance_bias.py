# -*- coding: utf-8 -*-
"""자모 위치별 '우연 일치' 수준 — 초성 오류율이 낮은 게 진짜 실력인가?"""
import random
SB,VC,TC=0xAC00,21,28; NC=VC*TC
def dec(s): return (s//NC,(s%NC)//TC,s%TC)

# 시나리오 1: 예측이 전체 음절공간에서 균등 랜덤
print("=== 시나리오 1: 완전 랜덤 예측 (11,172 균등) ===")
print(f"  초성 우연일치 = 1/19 = {1/19:6.1%}")
print(f"  중성 우연일치 = 1/21 = {1/21:6.1%}")
print(f"  종성 우연일치 = 1/28 = {1/28:6.1%}")

# 시나리오 2: 모델이 '흔한 자모'로 환각하는 경우 (현실적)
# 실사용 한국어에서 초성 ㅇ 약 20%, ㄱ 12%, ㅅ 9% / 종성 없음 약 55%
onset_prior = {11:0.20, 0:0.12, 9:0.09}   # ㅇ, ㄱ, ㅅ
final_prior_none = 0.55                     # 종성 없음 비율
print()
print("=== 시나리오 2: 흔한 자모 편중 환각 (현실적) ===")
print(f"  타깃 초성이 ㅇ일 확률 20% × 예측도 ㅇ일 확률 20% 등을 합산하면")
p_onset = sum(v*v for v in onset_prior.values()) + (1-sum(onset_prior.values()))**2/16
print(f"  초성 우연일치 ≈ {p_onset:6.1%}   (균등 랜덤 5.3%의 약 {p_onset/(1/19):.1f}배)")
p_final = final_prior_none**2 + (1-final_prior_none)**2/27
print(f"  종성 우연일치 ≈ {p_final:6.1%}   (균등 랜덤 3.6%의 약 {p_final/(1/28):.1f}배)")

print()
print("=== 시나리오 3: '받침 전부 탈락' 전략 모델 ===")
print("  종성 없는 음절만 골라 평가하면 종성 정확도 100%")
print(f"  전체 음절 중 종성 없음 = 399/11172 = {399/11172:.1%} (구조공간)")
print(f"  실사용 코퍼스 기준으론 약 {final_prior_none:.0%} → 겉보기 종성 정확도 {final_prior_none:.0%}")
print("  ⇒ 셀 균형 설계(§6.3) 없이는 이 전략이 고득점")

print()
print("=== 결론: 보정 지표 필요 ===")
print("  Chance-corrected Accuracy = (관측일치 - 우연일치) / (1 - 우연일치)")
for name, obs, ch in [("초성", 0.96, p_onset), ("중성", 0.80, 1/21), ("종성", 0.69, p_final)]:
    print(f"  {name}: 관측 {obs:.0%} → 보정 {(obs-ch)/(1-ch):6.1%}")
