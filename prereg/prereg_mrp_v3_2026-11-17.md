# Pré-registro — nova rodada do MRP com a dinâmica corrigida (v3)

Escrito em 2026-11-17, ANTES de lançar qualquer job de treino da rodada v3.

## Motivo
A dinâmica original do MRP (env.py e mrp_v2/mrp_env.py, "legacy") tem um estado absorvente: uma variável com |v| <= 1e-3 não recebe depleção nem correção e fica no alvo para sempre. Sob política aleatória, ~90% dos passos-variável ficam presos em C1/C2 (34% C5/C6, 11% C3/C4, 3% C7/C8). Correção v3: a depleção sempre se aplica (dentro da faixa segue o último sinal significativo); a correção não muda.

## Verificação feita antes das rodadas (não é predição)
`mrp_v3_check.py` → `docs/results_v3_mrp/analysis/V3_dynamics_check.csv`: com v3, a fração de passos-variável na faixa cai para 1–7% e nenhuma variável fica mais de 31 passos seguidos na faixa. Políticas de referência: C1/C2 mudam muito (aleatória 465 → 211; uniforme 884 → 228); C3–C8 praticamente não mudam (C5/C6 uniforme 89,6 → 88,0; o resto igual).

## O que não muda (fixado antes)
Condições C1–C8, variantes (sem OR adaptativo), 10 seeds de treino (42–51), 20 episódios de avaliação (1000–1019), hiperparâmetros padrão de runner.py, 10^5 passos (3·10^5 no subconjunto L2), desfechos primários (RMST com τ = 1000 com término; fração viável sem término), famílias de contraste e correção de Holm dos scripts analyze.py / analyze_nt.py, sem modificação.

## Predições (baseadas nos resultados legacy, docs/results_v2_noadapt/analysis)
- P1. O MLP 2×64 original continua sem aprender: em C3–C8 nenhuma variante fica significativamente acima da alocação uniforme.
- P2. O ator equivariante aprende: OR, DQ e Alive acima da uniforme em C3–C6.
- P3. OR > DR em C3–C8 (p_Holm < 0,05).
- P4. OR vs Alive: sem diferença significativa em C3–C8.
- P5. DR + alive > DR (o deslocamento de sinal ajuda o DR).
- P6. Geometria (C3–C6): OR > OR-min e OR-softmin; OR > OR sem zona em C3/C4; OR ≈ OR sem penalidade.
- P7. Sem término (C3–C8): OR > DR, dead-zone e OR sem zona em C3–C6; OR vs DQ sem diferença.
- P8. C1/C2 (exploratório: com legacy estavam no teto, então não há base para predição direcional). Expectativa fraca: a mesma ordem de C5/C6 (OR, DQ, Alive > DR).

Qualquer resultado será relatado, inclusive os que contrariem P1–P8.
