# BytePlus 서면 문의 초안

*설계서 v5.1 §16("약관 — aggregate 지표 공표 여부")이 요구한 서면 질문
템플릿을, 2026-08-12 약관 리스크 검토(법률 자문 아님) 결과를 반영해
실제 발송 가능한 형태로 작성했다.*

---

## 조사 결과 요약

BytePlus 문서 사이트(`docs.byteplus.com`)의 약관 페이지들은 자동 도구로는
**대부분 목차만 반환**되고 본문이 안 읽혀서, 웹 조사만으로는 최종 판단을
못 내렸다. 다만 조항 유형은 대략 파악됐다:

- **Output 소유권**: BytePlus는 소유권을 강하게 주장하지 않는 방향으로
  보인다("BytePlus does not claim ownership of the Output"). 단, 소유권
  비주장이 "무제한 재배포 허용"을 뜻하지는 않는다 — 사용은 여전히
  Agreement·ModelArk Terms 준수 조건에 묶여 있다.
- **학습/알고리즘 개발 금지 조항 (가장 중요)**: Model Services 약관에
  Output·Input·synthesized data를 모델/알고리즘의
  development·training·annotation·fine-tuning·optimization·iteration에
  쓰는 것을 금지하고, 제3자가 그렇게 쓰도록 허용하는 것도 금지하는
  구조가 있는 것으로 보인다. **CC BY 4.0은 이런 사용을 막지 않으므로
  가장 큰 리스크 지점이다.**
- **"vulnerability" 공개 조항**: 서비스 취약점 관련 정보를 제3자
  플랫폼에 공개하지 말라는 조항이 있는 것으로 보인다. 이 노트는 보안
  취약점이 아니라 측정 타당성 문제를 다루지만, 특정 모델의 실패율
  (예: "21.7% invalid glyphs")을 보고하므로 넓게 해석될 여지가 있다.
- **Trial/Free-Token 결합 시 벤치마크 공개 제한**: Customer Agreement의
  Trial Services 조항에 평가/벤치마크 결과 공개에 사전 서면승인을
  요구하는 문구가 있는 것으로 보인다. 유료 API 사용이면 리스크가
  낮아지나, 계정이 실제로 Trial/Free-Token 상태였는지는 별도 확인이
  필요하다.
- **Playground vs API**: Playground 전용 약관은 더 엄격한(private/
  non-commercial 지향) 문구를 포함하는 것으로 보인다. API로 생성했다면
  직접 적용은 약할 수 있으나, 계정 이력에 Playground/Free Token이
  섞여 있으면 리스크가 커진다.
- **미국 서비스 가용성**: 일부 문서상 BytePlus Model Services가 미국에서
  이용 불가능하다는 문구가 확인된다. 계정/이용자 소재지에 따라 확인이
  필요하다.
- **AI 생성 식별자 보존 의무**: GenAI AUP가 AI 생성 콘텐츠의 워터마크·
  메타데이터·식별자 제거를 금지하는 것으로 보인다.

**결론: 웹 조사만으로는 최종 판단이 불가능하다.** 아래 문의를 BytePlus에
직접 보내는 것이 유일하게 확실한 경로다. 회신 전까지 `docs/RELEASE.md`의
정책(이미지 미공개, 노트 본문 모델명 익명화)을 그대로 유지한다.

---

## 문의 초안 (영문, 그대로 발송 가능)

**수신:** BytePlus 고객지원 / ModelArk 지원팀
**제목:** Clarification on publication and redistribution of ModelArk/Seedream-generated outputs for academic technical note

> Hello BytePlus ModelArk Support,
>
> I am preparing a non-commercial, academic-style technical note on
> OCR-based measurement validity for visual text rendering. The study uses
> 300 images generated via the ModelArk API using the Seedream image
> generation model `dola-seedream-5-0-pro-260628`. The images are
> single-syllable Hangul text renderings and do not contain people,
> trademarks, private data, or third-party input images.
>
> Before any public release, I would like to confirm the governing terms
> and the permitted scope of publication.
>
> **1. Governing terms.** Could you confirm which terms govern this use
> case? Paid ModelArk API access; Seedream image generation model; not
> generated through the Playground UI; not intended for commercial product
> use. Specifically, is this governed by the "Specific Terms for the
> BytePlus Model Services", or by another Seedream/image-generation-specific
> agreement?
>
> **2. Aggregate results.** May we publish aggregate research results
> derived from the generated images, without releasing the raw images?
> Examples: accuracy tables, OCR confusion matrices, counts of
> malformed/non-typeable glyphs, statistical confidence intervals, error
> analysis.
>
> **3. Benchmark / vulnerability clause.** The note reports
> measurement-validity findings and is not intended as a model ranking or
> competitive benchmark. However, it necessarily reports some failure rates
> of generated Hangul text. Would BytePlus consider this publication to be
> a prohibited disclosure of "vulnerability" information, or a
> benchmark/evaluation result requiring prior written approval?
>
> **4. Raw image redistribution.** May we redistribute the 300 raw
> generated images for reproducibility of the technical note? If yes, under
> what conditions — is CC BY 4.0 permitted? If not, would a restricted
> research-evaluation license be acceptable, one that prohibits use of the
> images for training, fine-tuning, optimization, annotation, or
> development of models or algorithms, in order to comply with BytePlus
> Model Services terms?
>
> **5. Human labels and derived data.** May we release human labels, OCR
> outputs, filenames, hashes, and aggregate tables derived from the images,
> while withholding the raw images? (This is our current default and we
> believe it to be lower-risk, but would appreciate confirmation.)
>
> **6. Naming and attribution.** May the technical note identify the
> generating service as BytePlus ModelArk/Seedream and the model ID
> `dola-seedream-5-0-pro-260628` for reproducibility? If yes, what
> attribution or disclaimer should we include to make clear that BytePlus
> did not sponsor, review, or endorse the work?
>
> **7. Territory / account status.** Could you confirm that our account's
> paid API access to this Seedream model was permitted for our account and
> region, and whether any U.S. availability restrictions affect publication
> or redistribution of the generated outputs? Could you also confirm
> whether our account/order was ever classified as a Trial Service or under
> the Free-Token Campaign, since I understand that may require separate
> approval for publishing evaluation results?
>
> **8. AI-generated identifiers.** Are there any watermarks, metadata,
> identifiers, or AI-generated-content notices that must be preserved when
> storing or sharing the generated images?
>
> Our account email is: [계정 이메일]
>
> Thank you for your time. We will withhold the raw generated images, and
> withhold the generating vendor/model name from the public text of the
> note, until we receive written clarification.

---

## 발송 전 확인할 것

- [ ] 계정 이메일로 실제 문의 채널 확인 (콘솔 내 지원 티켓 / 공식 지원 이메일 등)
- [ ] 계정이 Trial/Free-Token 상태였는지 결제 이력에서 먼저 확인(문의문 §7에 넣을 근거)
- [ ] 회신 대기 중에는 `docs/RELEASE.md`의 정책(이미지 미공개, 노트 본문 모델명 익명화) 그대로 유지
- [ ] 회신 받으면 `docs/RELEASE.md`·`docs/DATASET_CARD.md`·`docs/TECHNICAL_NOTE(.ko).md` 갱신 + PDF 재생성

## 회신에 따른 처리

| 회신 | 처리 |
|---|---|
| 집계 통계만 허용, 원본 이미지·모델명 불가 | 현 상태(모델명 익명화 + 이미지 미공개) 그대로 유지 — 변경 없음 |
| 원본 이미지도 제한적 라이선스로 공개 가능 | `results/pilot/images/` 300장을 릴리스에 추가, `docs/RELEASE.md`의 "제한적 연구 라이선스" 조건 그대로 적용(학습/fine-tuning 금지 명시), 데이터셋 카드 갱신 |
| 원본 이미지 CC BY 4.0도 명시적으로 허용 | 드물게 유리한 경우 — 그래도 학습 금지 조건은 별도 계약 조항이므로, CC BY 4.0 라이선스 문구와 "허용된 용도" 고지를 병기하는 방식 검토 |
| 모델명 공표 허용 | 노트·카드에 모델명 복원 + 비후원 고지("BytePlus did not sponsor, review, or endorse...") 추가 |
| 집계 통계도 제한적 | 노트에서 구체 수치를 범주화된 구간(예: "70~80%")으로 대체하는 방안 검토 |
| Trial/벤치마크 조항 해당, 사전 승인 필요 | 별도 승인 절차 진행 — 회신 대기 |
| 응답 없음(N주 이상, 예: 4주) | 미공개·익명화 기본값 유지한 채로 공개 진행 — "확인 요청함, 응답 대기 중"으로 카드·노트에 명시 |
