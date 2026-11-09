# MRP — relatório final da revisão dos experimentos (AB-26-0360)

Data: 2026-11-02. Código: `MRP/mrp_v2/`. Resultados brutos: `MRP/results_v2/<estudo>/*.json` (4 892 rodadas de treino). Tabelas completas: `results_v2/analysis/RESULTS.md` e CSVs; figuras: `results_v2/analysis/figs/`.

Notação: **RMST** = tempo médio de sobrevivência restrito a τ = 1000 passos; unidade estatística = **seed de treino** (10 seeds, 42–51); cada seed é avaliada nos mesmos 20 episódios pareados (seeds 1000–1019). Testes: Friedman (com W de Kendall) por condição; Wilcoxon exato pareado por seed; p bruto, Holm e Bonferroni dentro da condição, e Holm global na CSV. Com 10 seeds o menor p exato possível é 0,002 (bicaudal); com 5 seeds é 0,0625 (por isso resultados com 5 seeds são apenas descritivos).

Condições: C1 (n=50, σ=0, b=1), C2 (50, 0,2, 1), C3 (50, 0, 0,2), C4 (50, 0,2, 0,2), C5 (100, 0, 1), C6 (100, 0,2, 1), C7 (100, 0, 0,2), C8 (100, 0,2, 0,2).

---

## 1. Conclusão em uma página

1. **O agente do artigo não aprende o MRP.** Com a rede original (MLP 2×64 sobre as 4n+1 entradas), todos os agentes treinados, com qualquer recompensa, terminam **abaixo** da política de alocação uniforme — que é exatamente o que a rede sem treino faz quando avaliada de forma determinística (seção A.5: C1 uniforme = 884 vs. treinados 622–718; C5 = 90 vs. 77–82; p_holm ≤ 0,012 em C1–C6). O comprimento dos episódios de treino não sobe. Os resultados antigos (SB3) mostram o mesmo padrão. **Portanto as Tabelas 5–7 do artigo comparam políticas que não aprenderam**; diferenças entre recompensas nessas tabelas são ruído de inicialização/seed.
2. **Com um ator que aprende** (mesma MLP pequena aplicada a cada variável, pesos compartilhados — "permutation-equivariant"), todas as recompensas positivas superam a uniforme e a alocação passa a ir para as variáveis mais desviadas (corr(alocação, desvio) de ≈ −0,1 para 0,6–0,9).
3. **Com esse agente, OR ≫ DR**, em todas as condições C3–C8 (p_holm = 0,012), e OR ≥ DQ (significativo em C4, C6 a 100k; em C3 e C5 a 300k).
4. **Mas OR não supera o controle de bônus de sobrevivência.** OR vs. "Alive" (r = +1 constante) não é significativo em nenhuma condição, a 100k ou 300k passos, com ou sem normalização da recompensa. Somar +1 ao DR (DR+alive) recupera praticamente toda a diferença (C5, 300k: DR = 88, DR+alive = 1000 em todas as 10 seeds, OR fixo = 993). **A vantagem do OR sobre o DR vem do sinal da recompensa, não da geometria de regimes.** Isso confirma a suspeita do revisor.
5. **Geometria:** dentro da família OR, a *zona de segurança* e a *agregação pela média* importam (OR fixo > OR sem zona, > min, > soft-min, > dead-zone; p_holm = 0,014 em C3–C6), mas a *penalidade assimétrica* não (OR fixo ≈ OR sem penalidade, n.s. em todas as condições). A zona importa sobretudo porque é ela que torna a recompensa positiva (OR sem zona tem média −0,15 por passo).
6. **Parâmetros adaptativos = deriva, não calibração.** OR adaptativo = OR fixo em todas as condições (nenhum contraste significativo). Partindo de 5 inicializações diferentes, z e β terminam perto de onde começaram (|Δz| ≤ 0,06, |Δβ| ≤ 0,32, contra diferenças de 0,4 em z₀ e 1,5 em β₀ entre as inicializações; P1). Quando se movem, movem-se para o lado "leniente" (z ↑, β ↓), como prevê o argumento de otimização do revisor.
7. **Ruído:** não piora nenhuma recompensa (nem com o agente que aprende, nem com σ até 0,4). O mecanismo "baselines sofrem com gradientes ruidosos" não tem suporte.
8. **Escassez (C3, C4, C7, C8):** com o MLP original todas as recompensas empatam no piso; com o ator que aprende, OR > DR, mas OR = Alive e, em C7/C8, DQ ≥ OR (diferenças < 1 passo).

---

## 2. Problemas encontrados no código e no texto antigos

| # | Problema | Efeito | Correção no v2 |
|---|---|---|---|
| 1 | 24 episódios por célula vinham de **3 seeds de treino** e eram tratados como independentes | pseudo-replicação; p-valores muito pequenos (ex.: OR vs Fixed +234, p < 0,001) | 10 seeds, teste no nível da seed |
| 2 | OR adaptativo começava em **β = softplus(0) = 0,693**; OR fixo usava **β = 1,0** | a ablação Adaptive vs Fixed confunde adaptação com valor inicial | ambos começam em z = 0,5, β = 1,0 |
| 3 | No pipeline SB3, z e β eram **saídas da rede de política (dependentes do estado)**; em `ppo.py` eram escalares globais (init β = softplus(1) ≈ 1,31) | o texto (Eqs. 18–19) não corresponde ao que foi rodado; a Tabela 8 reporta médias de amostras de saídas dependentes do estado | parâmetros escalares, independentes do estado, aprendidos pelo log-prob do PPO; tabela com valores brutos e transformados |
| 4 | Episódios de avaliação **não pareados** (ruído e passeio aleatório do decaimento compartilhavam o mesmo gerador com a política) | comparações entre métodos não viam o mesmo processo exógeno | geradores separados por episódio para dinâmica e ruído |
| 5 | Horizonte de **1001** passos (`step_count > max_steps`) | | truncamento em exatamente 1000 |
| 6 | O decaimento de cada variável **segue um passeio aleatório** dentro do episódio; o texto diz que é "held fixed within the episode" | descrição incorreta do ambiente | mantido o passeio (dinâmica idêntica); **corrigir o texto** |
| 7 | Ruído de observação somado a **todas as 4n+1 entradas** (inclusive decaimentos e orçamento); o texto (Eq. 28) diz só nas variáveis | | padrão: só nas variáveis; o original é testado em N3 (sem efeito relevante) |
| 8 | Arquitetura não aprende (seção 1, item 1) | todas as comparações do artigo são entre políticas não treinadas de fato | estágio L com ator compartilhado |
| 9 | Em `run_ablation_or_fixed_experiment.py` (e `run_full_mrp_experiment.py`), **só o OR adaptativo era treinado com o PPO do Stable-Baselines3**; OR fixo, DQ e DR usavam uma **implementação própria de PPO** | a ablação Adaptive vs Fixed (e OR vs DQ/DR) compara dois algoritmos diferentes, não só duas recompensas; o texto diz que todos usaram SB3 | um único PPO para todas as variantes |
| 10 | Com SB3, o espaço de ação `Box[0, 1]` faz o SB3 **cortar as ações em [0, 1]** antes do softmax; o PPO próprio não corta | no OR adaptativo a razão máxima entre alocações é e ≈ 2,7 (alocação quase uniforme, que é uma boa política sem aprender); nos demais as logits são livres | logits livres para todas as variantes |

---

## 3. Resposta ponto a ponto (MRP) — resultados

### 3.1 "Comparison with continuous baselines" (p-valores brutos vs. corrigidos)
Todas as tabelas novas trazem p bruto, Holm e Bonferroni (dentro da condição) e Holm global. Resultados principais com o ator que aprende (100k, 10 seeds, `L_table_main_contrasts.csv`):

| Cond | OR adapt. | OR fixo | DQ | DR | Friedman |
|---|---|---|---|---|---|
| C1 | 978 ± 39 | 981 ± 51 | 971 ± 38 | 930 ± 126 | p = 0,715 |
| C2 | 964 ± 46 | 952 ± 73 | 903 ± 70 | 893 ± 72 | p = 0,027 |
| C3 | 62,2 ± 3,5 | 62,7 ± 2,8 | 58,5 ± 5,5 | 49,2 ± 1,8 | p < 0,001, W = 0,71 |
| C4 | 61,4 ± 2,6 | 63,9 ± 3,4 | 60,6 ± 3,4 | 48,3 ± 2,0 | p < 0,001, W = 0,81 |
| C5 | 139 ± 32 | 169 ± 72 | 124 ± 19 | 89,5 ± 4,1 | p < 0,001, W = 0,72 |
| C6 | 138 ± 35 | 137 ± 30 | 111 ± 7,5 | 86,5 ± 4,8 | p < 0,001, W = 0,80 |
| C7 | 29,8 ± 0,4 | 30,1 ± 0,4 | 30,2 ± 0,3 | 29,0 ± 0,3 | p < 0,001, W = 0,78 |
| C8 | 29,7 ± 0,4 | 29,9 ± 0,5 | 30,1 ± 0,4 | 29,0 ± 0,2 | p < 0,001, W = 0,64 |

- OR (ambos) > DR em C3–C8: p_holm = 0,012 (C8 adaptativo: 0,023).
- OR vs DQ: OR fixo > DQ em C4 e C6 (p_holm = 0,012); OR adaptativo > DQ em C2 (0,047) e C6 (0,020); DQ > OR adaptativo em C8 (0,041; Δ = 0,5 passo). Demais n.s.
- Treino longo (300k, 10 seeds, C3 e C5): OR > DQ > DR (p_holm 0,020–0,023), exceto OR adaptativo vs DQ em C5 (n.s.).

### 3.2 "Replication unit and censoring"
Feito: 10 seeds, testes no nível da seed, RMST a τ = 1000, curvas de Kaplan–Meier (`figs/L_km_main.png`, `A_km_main.png`), fração que chega a τ (seção A).

### 3.3 "Reward sign and scale"
- **Sinal por passo** (mesmas trajetórias, política aleatória; seção R): OR fixo +0,80 (100 % positivo); DR −0,145 (100 % negativo); DQ +0,001; Alive +1; DR+alive +0,85.
- **Controles de bônus** (L.2, 100k): OR vs Alive n.s. em todas as condições após Holm (bruto 0,02–0,04 em C3); DR+alive > DR em C3–C8 (p_holm = 0,012); OR fixo vs DR+alive n.s. exceto C3 (+5,7, p_holm = 0,020).
- **300k** (L.7b): OR vs Alive n.s. (C3 +15 a +17, p = 0,28; C5 −18 a +37); DR+alive em C5 = 1000 ± 0.
- **Normalização da recompensa** (L.8; running-return como VecNormalize): o padrão não muda — OR = DQ = Alive = DR+alive > DR. A normalização não resgata o DR (C5: 89 → 93), porque não muda o sinal. **Logo, o que importa é o sinal/offset, não a escala.**
- **Tuning modesto** (T/F; lr × normalização, com o MLP original): nenhuma ordenação consistente; nenhuma vantagem do OR sobrevive à correção.

### 3.4 "Isolating the regime structure" (L.2b, ator que aprende, C3–C6)

| Cond | OR fixo | sem penalidade (β=0) | sem zona (z=0) | dead-zone | min | soft-min | Alive | DR |
|---|---|---|---|---|---|---|---|---|
| C3 | 62,7 | 61,9 | 56,8 | 52,6 | 46,8 | 49,2 | 59,3 | 49,2 |
| C4 | 63,9 | 60,7 | 58,5 | 50,6 | 47,3 | 49,4 | 62,2 | 48,3 |
| C5 | 168,9 | 131,5 | 71,9 | 110,1 | 75,2 | 100,8 | 138,0 | 89,5 |
| C6 | 137,0 | 124,8 | 74,7 | 97,4 | 75,2 | 92,2 | 139,0 | 86,5 |

- OR fixo > sem zona, > min, > soft-min: p_holm = 0,014 em todas; > dead-zone: p_holm 0,014–0,029.
- OR fixo vs sem penalidade: n.s. em todas (p_holm 0,17–0,56). OR sem penalidade vs Alive: n.s.
- Dead-zone > DR em C5 (p_holm = 0,014) e C6 (0,041).
- **Leitura:** a forma do OR é r_i = 1 se a variável está na zona *ou* melhorou; r_i = −β só se está fora da zona e piorou. Com a média sobre 50–100 variáveis isso é quase sempre ≈ +1, ou seja, um bônus de sobrevivência com uma pequena correção. Retirar a zona torna a recompensa negativa em média (−0,15) e derruba o desempenho; min/soft-min também tornam a recompensa negativa (−0,55/−0,20). A penalidade assimétrica não contribui.

### 3.5 "Interpretation of the adaptive parameters"
- **Inicialização** (agora explícita): μ_sz = 0 → z = σ(0)·0,8 + 0,1 = 0,50; μ_β = softplus⁻¹(1) = 0,5413 → β = 1,00; log σ = −1. OR fixo usa exatamente esses valores.
- **Adaptativo = Fixo**: nenhum contraste significativo (100k e 300k, com e sem normalização).
- **P1 (5 inicializações, MLP):** os valores finais ficam perto de onde começaram (|Δz| ≤ 0,06; |Δβ| ≤ 0,32; maior deriva em C7, z₀ = 0,3, β₀ = 2,0). Se fosse calibração, convergiriam.
- **P2/P3:** OR fixo nos valores aprendidos na própria condição ≈ OR fixo padrão ≈ OR fixo nos valores aprendidos na condição com orçamento trocado. Os valores aprendidos não são específicos do ambiente.
- **Com o ator que aprende (L.5):** z sobe e β cai levemente em C3, C4, C7, C8 (z 0,53–0,55, β 0,91–0,96; p_holm < 0,05) — direção "leniente", que aumenta a recompensa sem melhorar a sobrevivência. A 400k (MLP) a deriva continua (C7: z 0,59, β 0,83).
- **Tabela 8 antiga:** valores eram médias de saídas dependentes do estado (pipeline SB3) com β inicial 0,693; isso explica os números (β ≈ 0,57–0,77) e a discrepância com o texto. A nova tabela reporta μ bruto e valores transformados.

### 3.6 "Observation-noise mechanism"
- Recompensa e término usam o estado latente; o ruído afeta só a observação da política (padrão v2).
- N1 (MLP, σ ∈ {0; 0,1; 0,2; 0,4}): nenhuma recompensa degrada (Δ de −9 % a +8 %, ICs cruzando zero).
- N2 (recompensa calculada da observação ruidosa) e N3 (ruído em todas as entradas, como no código original): sem efeito após Holm.
- L.9 (ator que aprende, σ 0 → 0,2): nenhuma degradação significativa.
- **Conclusão:** o ganho do OR em C2 no artigo era variabilidade de seed (P2 mostra diferenças de ~100 passos em C1 só por seed).

### 3.7 "Scarcity claims"
Com o MLP original: nenhum Friedman significativo em C3, C4, C7, C8. Com o ator que aprende: OR > DR, mas OR = Alive e, em C7/C8, DQ tem a maior média (diferença < 1 passo). A alegação de vantagem do OR na escassez deve ser retirada ou restrita a "OR, DQ e Alive superam DR".

---

## 4. Texto sugerido (inglês) para o manuscrito

**Methods — agent architecture (new paragraph).**
> In preliminary analyses we found that a single multilayer perceptron over the full (4n+1)-dimensional observation does not learn the allocation task within the training budget: for every reward function, the trained deterministic policy performed at or below uniform allocation (the behaviour of the untrained network), and training-episode length did not increase (Supplementary Table S-A5). We therefore use a permutation-equivariant actor: a two-layer MLP (64 tanh units per layer) applied independently to each variable's features (v_i, |v_i|, 100·λ_i, b, mean_j |v_j|) produces that variable's allocation logit, and the allocation is b·softmax(logits). The critic, PPO hyperparameters, training budget (10⁵ steps; 3×10⁵ in a long-training check), training seeds and evaluation episodes are identical across reward functions.

**Methods — statistics.**
> Each reward function was trained with 10 independent seeds; each trained policy was evaluated on the same 20 evaluation episodes (identical initial states, decay processes and observation-noise sequences across methods). The unit of analysis is the training seed. The outcome is the restricted mean survival time (RMST) at τ = 1000 steps. We report Friedman tests (blocked by seed) with Kendall's W, and pairwise exact Wilcoxon signed-rank tests with raw, Holm- and Bonferroni-adjusted p-values (family = the contrasts within a condition). With 10 seeds the smallest attainable two-sided exact p is 0.002.

**Methods — adaptive parameters.**
> The safety-zone and penalty parameters are state-independent: z_safe = 0.8·σ(s_z) + 0.1 and β = softplus(s_β), with s_z ~ N(μ_sz, σ_sz) and s_β ~ N(μ_β, σ_β) sampled once per rollout step and optimised through the PPO likelihood ratio. We initialise μ_sz = 0 (z_safe = 0.50), μ_β = softplus⁻¹(1) = 0.541 (β = 1.00) and log σ = −1. OR-Fixed uses exactly these initial values. Note that the OR reward is a step function of the state and is not differentiated; the parameters are learned by the score-function (policy-gradient) estimator.

**Methods — environment correction.**
> Per-variable decay rates are drawn at reset and then follow a Gaussian random walk within the episode [replace "held fixed within the episode"]. Observation noise is added to the state variables only; reward and termination are computed from the latent state.

**Results — main comparison.**
> With an agent that learns, the positive-valued rewards (OR, DQ) clearly outperformed the squared-deviation drive DR in all conditions in which survival was below the ceiling (C3–C8; Holm-adjusted p = 0.012, Table X). Adaptive and Fixed OR did not differ in any condition. OR outperformed DQ in C4 and C6 (Fixed OR, p_Holm = 0.012) and, after 3×10⁵ steps, in C3 and C5 (p_Holm ≤ 0.023).

**Results — sign controls.**
> However, OR did not outperform a constant survival reward (r_t = +1; "Alive") in any condition (all p_Holm ≥ 0.078), and adding a constant +1 to DR ("DR + alive") recovered most of the gap to OR (e.g. C5 after 3×10⁵ steps: DR 88 ± 8, DR + alive 1000 ± 0, Fixed OR 993 ± 21 steps). The same pattern held with running-return reward normalisation, which removes scale differences but not sign. We conclude that, in the MRP, the advantage of OR over DR is explained primarily by the sign of the per-step reward (OR is positive on > 99% of steps), not by its regime structure.

**Results — regime geometry.**
> Within the OR family, removing the safety zone (z = 0) or replacing the mean by a min or soft-min aggregation reduced survival (all p_Holm = 0.014 in C3–C6), whereas removing the asymmetric penalty (β = 0) had no detectable effect. All three harmful modifications make the expected per-step reward negative, consistent with the sign account above.

**Results — adaptive parameters.**
> The learned parameters remained close to their initial values. From five different initialisations, final z_safe and β stayed within 0.06 and 0.32 of their starting points (initial values spanned 0.3–0.7 and 0.5–2.0), and fixing OR at the learned values (own condition or budget-swapped condition) did not change performance. Where the parameters moved, they moved in the lenient direction (larger z_safe, smaller β), which increases the reward without improving survival. We therefore interpret these changes as optimisation-driven drift rather than environment-specific calibration.

**Discussion — wording to remove or change.**
- "differentiable regime-based objective" → "regime-based reward whose parameters are learned by policy gradient".
- "asymmetric penalty": with β < 1 the penalty is *smaller* than the reward; describe β as "penalty magnitude".
- Remove "OR significantly outperforms DQ and DR in C5" (old Table 7) and the scarcity claims for C3/C4/C7/C8; replace by the new results.
- Remove the noise mechanism ("baselines suffer from noisy gradients"); state that no reward degraded under observation noise.
- Label every p-value as raw or adjusted.

---

## 5. Texto sugerido (inglês) para a resposta aos revisores

> **Comparison with continuous baselines / p-values.** We thank the reviewer. All MRP experiments were rerun (Section X). Every table now reports raw, Holm- and Bonferroni-adjusted p-values (family = contrasts within a condition) and the claims in the text are based on adjusted values only. The previous statement that OR "significantly outperforms" DQ and DR in C5 has been removed.

> **Replication unit and censoring.** We now use 10 training seeds per reward function and condition, test at the seed level (Friedman and exact Wilcoxon signed-rank tests, n = 10), and use the restricted mean survival time at τ = 1000 together with Kaplan–Meier curves. We also corrected the horizon (the previous code allowed 1001 steps) and made evaluation episodes exactly paired across methods.

> **Learnability (new).** While rerunning the experiments we found that the original agent (one MLP over the full observation) did not learn the MRP: for every reward function the trained policy performed at or below uniform allocation, which is the behaviour of the untrained network. The previous MRP comparisons were therefore between effectively untrained policies, and we have withdrawn them. The new experiments use a permutation-equivariant actor that does learn (it improves on uniform allocation by up to 79 steps in C5 and directs resources to the most deviated variables).

> **Reward sign and scale.** The reviewer's concern is confirmed. We added a constant survival reward (Alive, +1), DR + alive and DQ + alive, reward normalisation, and a modest per-method tuning grid. OR outperforms DR, but it does not outperform the Alive control in any condition, and DR + alive closes the gap with OR. The result is unchanged with reward normalisation and after 3×10⁵ training steps. We now state that, in the MRP, OR's advantage over DR is attributable mainly to its positive per-step sign.

> **Adaptive parameters.** We now state the initialisation (μ_sz = 0, z_safe = 0.50; μ_β = 0.541, β = 1.00; OR-Fixed uses the same values; previously, Adaptive OR started at β = 0.693 while OR-Fixed used β = 1.0, which confounded the ablation). Table 8 now reports both the raw means and the transformed values. To distinguish drift from calibration we (i) started from five different initialisations, (ii) fixed OR at the values learned in the same condition and (iii) in a condition with a different budget. The parameters stay near their initial values, Fixed OR at learned values performs the same as default Fixed OR, and Adaptive and Fixed OR do not differ. We therefore no longer claim environment-specific calibration, and we have replaced "differentiable regime-based objective".

> **Observation-noise mechanism.** Yes: reward and termination are computed from the latent state and noise affects only the policy's observation. A noise sweep (σ = 0–0.4) and two direct manipulations (reward computed from the noisy observation; noise on all inputs) show no degradation for any reward function; the C2 difference in the original manuscript was within seed variability. We have removed the proposed mechanism.

> **Scarcity claims.** With 10 seeds, OR, DQ and Alive all outperform DR in the scarce conditions, but OR does not outperform DQ or Alive (in C7/C8 DQ has the highest mean, by < 1 step). The Discussion has been reworded accordingly.

> **Isolating the regime structure.** We added the requested controls (dead-zone drive, OR without safety zone, OR without asymmetric penalty, min and soft-min aggregation). Removing the safety zone or using min/soft-min aggregation reduces survival; removing the penalty has no effect. All harmful modifications make the per-step reward negative on average, so these results are consistent with the sign explanation rather than with a specific contribution of the three-regime partition.

---

## 6. Figuras e tabelas para usar

| Uso | Arquivo |
|---|---|
| Não aprendizado do agente original | `figs/A_bounds.png`, `A_bounds.csv`, `figs/A_learning_main.png` |
| Resultado principal (ator que aprende) | `figs/L_km_main.png`, `figs/L_forest_main.png`, `L_table_rmst.csv`, `L_table_main_contrasts.csv` |
| Controles de sinal | `figs/L_forest_sign.png`, `L_table_sign_controls.csv` |
| Geometria | `figs/L_forest_geometry.png`, `L_table_geometry_controls.csv` |
| Treino longo | `figs/L_forest_long300k.png`, `L_table_long300k_contrasts.csv` |
| Normalização | `figs/L_forest_rewnorm.png`, `L_table_rewnorm_contrasts.csv` |
| Parâmetros adaptativos | `figs/P1_init_to_final.png`, `P1_inits_final_params.csv`, `P2_fixed_at_learned.csv`, `L_learned_params.csv` |
| Ruído | `figs/N1_noise_sweep.png`, `N2_N3_manipulations.csv`, `L_table_noise_effect.csv` |
| Arquitetura | `L_table_arch_effect.csv`, `L_behaviour.csv` |

## 7. Limitações e pontos em aberto

- Os resultados com 300k passos só cobrem C3 e C5 (C1 com 5 seeds, descritivo). Em C3 nenhuma recompensa chega perto do oráculo (325 passos); o agente ainda está longe do ótimo.
- Os controles de geometria e de normalização foram rodados só em C3–C6 (em C1/C2 tudo fica perto do teto; em C7/C8, perto do piso).
- O tuning (T/F) foi feito com o MLP original, que não aprende; não foi repetido com o ator compartilhado.
- A seção Crafter não foi tratada aqui (fora do escopo de `entradas` para o MRP).
- Implicação para o artigo: no MRP, a afirmação central sustentável é "recompensas positivas por passo (OR, Alive, DR+alive) superam a penalidade quadrática DR"; **não** há evidência de que a geometria de três regimes ou a adaptação de z/β acrescentem algo além do sinal.
