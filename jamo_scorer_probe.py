# -*- coding: utf-8 -*-
"""자모 단위 채점 규칙의 실측 검증 — 어디까지 정확히 잴 수 있는가"""
SB, VC, TC = 0xAC00, 21, 28
NC = VC*TC
L="ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
V="ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
T="_ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

def dec(ch):
    s=ord(ch)-SB
    if not (0<=s<11172): return None
    return (L[s//NC], V[(s%NC)//TC], T[s%TC])

def score(target, pred):
    """반환: (판정유형, 초성ok, 중성ok, 종성ok)"""
    t=dec(target)
    if pred is None or pred=="":       return ("EMPTY", None,None,None)
    p_valid=[c for c in pred if dec(c)]
    if len(p_valid)==0:                return ("NON_HANGUL", None,None,None)
    if len(pred)>1:                    return ("OVERGEN", None,None,None)  # 길이 불일치
    p=dec(pred)
    return ("VALID", t[0]==p[0], t[1]==p[1], t[2]==p[2])

cases=[
 ("읽","엮","파일럿 실측: 겹받침 붕괴"),
 ("책","챽","파일럿 실측: 중성 오류"),
 ("일","엄","파일럿 실측: 중성+종성"),
 ("입","업","파일럿 실측: 중성만"),
 ("가","가","정답"),
 ("이","익","종성 삽입(원래 없음)"),
 ("읽","익","겹받침→단순종성"),
 ("읽","읽기","과생성 2글자"),
 ("읽","","빈 출력"),
 ("읽","ag","비한글"),
 ("읽","ㅇㅣㄺ","자모 분리 출력"),
 ("의","이","복합모음→단순"),
 ("값","갑","겹받침 일부 탈락"),
]
print(f"{'타깃':<4}{'예측':<8}{'판정':<12}{'초':<4}{'중':<4}{'종':<4}  설명")
print("-"*68)
for t,p,d in cases:
    r=score(t,p)
    f=lambda x: "-" if x is None else ("O" if x else "X")
    print(f"{t:<4}{p if p else '(빈)':<8}{r[0]:<12}{f(r[1]):<4}{f(r[2]):<4}{f(r[3]):<4}  {d}")
