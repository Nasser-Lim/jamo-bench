# 데이터 파티션 단일 스펙

*JAMO_v51_patch.md §19의 실행 순서를 그대로 옮기고, 각 단계와 `jamo_bench.partitioning`의
함수를 매핑한다. 이 순서를 바꾸면 오염이 난다 — reserve 확보 전에 private-test holdout을
뽑으면 최소쌍 사다리가 holdout으로 새어 나갈 수 있고, holdout 전에 Core를 확정하면
private-test 전용 음절이 하나도 남지 않을 수 있다.*

## 실행 순서

| # | 단계 | 구현 | 비고 |
|---|---|---|---|
| 1 | 금칙어 목록 제외 | `partition(banned_syllables=...)` | 실제 비속어·혐오표현 목록은 운영자가 채운다(§17 윤리). 기본값은 빈 집합. 제외 목록 해시만 공개, 셀별 제외 개수 표 게시(v5.1 §19). |
| 2 | Minimal-pair reserve 확보 | `minimal_pair_reserve_map()` | 종성 사다리(가/각/간/갈/감/갑/값), 종성 복잡도(일/입/잉/읽), 복합모음(와/워/왜/의/외/위), 쌍받침 대조(각/갂, 갓/갔) — v5.1 §2. |
| 3 | private-test holdout 분리 (구조 층화) | `partition()` 내부, 18셀 균등 `private_holdout_per_cell`장 | reserve 문자는 holdout 후보에서 제외 — Core에 남아야 하는 확정 항목이므로. |
| 4 | Core 540 확정 (18셀×30, tensed_double·vertical_derived 층화 포함) | `_stratified_cell_sample()` | `simple_T` 셀은 tensed_double(ㄲㅆ 받침) ≥20%(v5.1 §3), `simple_V` 셀은 vertical_derived(ㅐㅒㅔㅖ) 30~40%(v5.1 §4). |
| 5 | Cross-shared 250 ⊂ Core 540 | `partition()` 내부 | 18셀 층화, 셀당 base 13~14개, 250에 맞춰 조정(v5.1 §1). |
| 6 | Cross-Struct 180 ⊂ Core 540 | `partition()` 내부 | 6셀 = 2모음×3종성(초성군 접음), 셀당 30개(v5.1 §12, §20-25). |
| 7 | Word-Freq 125 / Word-Struct 125 | `build_word_subsets()` (스텁) | 실제 한국어 단어 빈도 코퍼스 필요 — 이번 스텝 범위 밖. |
| 8 | Forge-train 가능 풀 = 11,172 − private-test | *(다음 스텝)* | Forge 렌더러 구현 시 `PartitionResult.private_test_holdout`을 전체 11,172에서 제외하면 된다. |
| 9 | seen / unseen-structure split | *(다음 스텝)* | Forge 효과 실험(§7.5)에서 구현. |

**30 미만 셀이 2개 이상 발생 → Plan B' 트리거.** `PartitionResult.plan_b_prime_triggered`로 노출되며,
`cell_available_counts`에 셀별 실제 가용 후보 수가 함께 기록된다(§6.5 폴백 사다리).

## 재현성

`partition(seed=N)`은 같은 `N`에 대해 항상 같은 결과를 낸다(`numpy.random.default_rng(seed)` 고정).
결과는 `PartitionResult.to_json()` / `to_json_str()`으로 직렬화해 리포지토리에 커밋한다 — 실제
이미지 생성 파이프라인은 이 JSON을 그대로 소비한다.

## 이번 스텝에서 다루지 않은 것

- **Word-Freq/Word-Struct**: 실제 한국어 단어 빈도 코퍼스가 있어야 정의 가능. 코퍼스 확보 후
  `build_word_subsets()`를 채운다.
- **금칙어 목록 자체**: 비속어·혐오표현 조합 사전 제외 목록(§17)은 운영자가 결정할 콘텐츠
  정책이므로 라이브러리가 임의로 채우지 않는다.
- **private-test holdout 크기**: `private_holdout_per_cell` 기본값(5)은 조정 가능한 파라미터로만
  남겨둔다 — 설계서가 셀당 정확한 개수를 못박지 않았다.
