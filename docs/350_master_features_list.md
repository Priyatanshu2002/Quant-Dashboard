# Master 350 Feature Inventory for Quantum Machine Learning Futures Model

This document provides the complete, perfectly sorted inventory of **350 continuous, stationary features** organized into **10 domain modules** across 7 futures contracts (`ES`, `NQ`, `ZN`, `CL`, `DX`, `6E`, `GC`) and underlying index constituents for Quantum Feature Selection (QFS) and machine learning model benchmarking.

---

## Module 1: Classical Trend, Momentum & Technical Oscillators (Features 1 – 45)
1. **`rsi_14`**: 14-bar Relative Strength Index (RSI)
2. **`macd_line`**: Moving Average Convergence Divergence (MACD) line
3. **`macd_signal`**: MACD signal line
4. **`macd_hist`**: MACD histogram difference
5. **`adx_14`**: 14-bar Average Directional Index (ADX) measuring trend strength
6. **`stochastic_k_14`**: 14-bar Stochastic %K oscillator
7. **`stochastic_d_14`**: 14-bar Stochastic %D moving average
8. **`williams_r_14`**: 14-bar Williams %R momentum oscillator
9. **`cci_20`**: 20-bar Commodity Channel Index (CCI)
10. **`aroon_up_25`**: 25-bar Aroon Up indicator
11. **`aroon_down_25`**: 25-bar Aroon Down indicator
12. **`aroon_oscillator_25`**: 25-bar Aroon Oscillator difference
13. **`rate_of_change_roc_12`**: 12-bar Rate of Change (ROC) percentage
14. **`chande_momentum_oscillator_14`**: 14-bar Chande Momentum Oscillator (CMO)
15. **`stoch_rsi_k_14`**: 14-bar Stochastic RSI %K oscillator
16. **`stoch_rsi_d_14`**: 14-bar Stochastic RSI %D moving average
17. **`connors_rsi_3_2_100`**: 3-component Connors RSI (RSI 3, Streak RSI 2, Percent Rank 100)
18. **`supertrend_direction_14`**: SuperTrend indicator directional state (+1 Bullish, -1 Bearish)
19. **`supertrend_distance_ratio`**: Stationary distance ratio of price to active SuperTrend trailing band
20. **`fisher_transform_14`**: Ehlers Gaussian Fisher Transform of normalized price
21. **`fisher_transform_signal`**: Signal line of Ehlers Fisher Transform
22. **`center_of_gravity_oscillator`**: Ehlers Center of Gravity (CG) oscillator measuring cycle pivots
23. **`ehlers_decycler_distance`**: Stationary distance ratio to Ehlers Zero-Lag Decycler trend line
24. **`ehlers_sine_wave_phase`**: Ehlers Sine Wave phase angle distinguishing cycle vs trend modes
25. **`coppock_curve`**: Coppock Curve multi-period momentum oscillator
26. **`know_sure_thing_kst`**: KST multi-timeframe momentum oscillator
27. **`trix_15`**: 15-bar Triple Exponentially Smoothed Oscillator (TRIX)
28. **`detrended_price_osc_20`**: 20-bar Detrended Price Oscillator (DPO)
29. **`schaff_trend_cycle_20`**: 20-bar Schaff Trend Cycle MACD oscillator
30. **`ultimate_oscillator`**: Multi-timeframe Ultimate Oscillator
31. **`log_return_1bar`**: 1-minute stationary log return
32. **`log_return_5bar`**: 5-minute stationary log return
33. **`log_return_15bar`**: 15-minute stationary log return
34. **`log_return_30bar`**: 30-minute stationary log return
35. **`log_return_60bar`**: 1-hour stationary log return
36. **`log_return_120bar`**: 2-hour stationary log return
37. **`return_globex_open`**: Log return since Globex 18:00 EST session open (session-reset)
38. **`return_4h`**: 4-hour stationary log return
39. **`ema_ratio_5m_1h`**: Ratio of 5-min EMA to 1-hr EMA
40. **`linreg_slope_20`**: 20-bar linear regression price slope
41. **`close_location_value`**: Close location value relative to high-low bar range
42. **`position_in_session_range`**: Relative location within current session high-low range
43. **`opening_drive_strength`**: Globex open momentum impulse
44. **`choppiness_index_14`**: 14-bar Choppiness Index (trending vs consolidation)
45. **`mass_index_25`**: 25-bar Mass Index (reversal squeeze indicator)

---

## Module 2: Realized & Implied Volatility, Squeezes & Fractals (Features 46 – 80)
46. **`atr_14`**: 14-bar Average True Range (ATR)
47. **`atr_ratio_5m_4h`**: Ratio of 5-min ATR to 4-hr ATR
48. **`atr_ratio_15m_1d`**: Ratio of 15-minute ATR to Daily ATR
49. **`realized_vol_15`**: 15-bar realized volatility
50. **`realized_vol_60`**: 60-bar realized volatility
51. **`parkinson_vol_20`**: 20-bar Parkinson high-low volatility
52. **`parkinson_vol_60`**: 60-bar Parkinson high-low volatility
53. **`garman_klass_vol_20`**: 20-bar Garman-Klass OHLC volatility
54. **`garman_klass_vol_60`**: 60-bar Garman-Klass OHLC volatility
55. **`yang_zhang_vol_20`**: 20-bar Yang-Zhang overnight+intraday volatility
56. **`yang_zhang_vol_60`**: 60-bar Yang-Zhang overnight+intraday volatility
57. **`rogers_satchell_vol_20`**: 20-bar Rogers-Satchell non-zero drift volatility
58. **`rogers_satchell_vol_60`**: 60-bar Rogers-Satchell non-zero drift volatility
59. **`parkinson_ratio_20_60`**: Ratio of 20-bar to 60-bar Parkinson volatility
60. **`garman_klass_ratio_20_60`**: Ratio of 20-bar to 60-bar Garman-Klass volatility
61. **`garch_cond_vol`**: Maximum-Likelihood GARCH(1,1) conditional volatility
62. **`chaikin_volatility_20`**: 20-bar Chaikin High-Low volatility expansion
63. **`bollinger_bandwidth_20`**: 20-bar Bollinger Bandwidth (squeeze indicator)
64. **`bollinger_percent_b_20`**: 20-bar Bollinger %B oscillator
65. **`keltner_channel_width_20`**: 20-bar Keltner Channel bandwidth
66. **`donchian_channel_width_20`**: 20-bar Donchian High-Low channel width
67. **`donchian_percent_b_20`**: Donchian Channel %B relative location score
68. **`historical_vol_ratio_5_20`**: Ratio of 5-bar to 20-bar historical volatility
69. **`historical_vol_ratio_10_60`**: Ratio of 10-bar to 60-bar historical volatility
70. **`volatility_of_volatility_60`**: Standard deviation of rolling 20-bar volatility
71. **`return_skewness_60`**: 60-bar return distribution skewness
72. **`realized_kurtosis_60`**: 60-bar return distribution realized kurtosis (fat tails)
73. **`downside_semivariance_60`**: 60-bar downside semivariance (downside risk)
74. **`upside_semivariance_60`**: 60-bar upside semivariance (upside potential)
75. **`sortino_ratio_proxy_60`**: 60-bar rolling return to downside semivariance ratio
76. **`kaufman_efficiency_ratio_20`**: 20-bar Kaufman Efficiency Ratio (trending vs noise)
77. **`kaufman_efficiency_ratio_60`**: 60-bar Kaufman Efficiency Ratio
78. **`variance_ratio_test_15_60`**: Variance ratio test for random walk vs trend
79. **`hurst_exponent_100`**: Rescaled range Hurst exponent over 100 bars
80. **`dfa_exponent_100`**: Detrended Fluctuation Analysis fractal exponent

---

## Module 3: Anchored VWAP, TWAP & Market Profile Pivots (Features 81 – 110)
81. **`dist_session_vwap`**: Distance ratio to current Anchored Session VWAP
82. **`dist_avwap_swing_high`**: Distance ratio to Anchored VWAP from recent swing high
83. **`dist_avwap_swing_low`**: Distance ratio to Anchored VWAP from recent swing low
84. **`avwap_spread`**: Spread between Swing High AVWAP and Swing Low AVWAP
85. **`anchored_vwap_dev_upper_1s`**: Distance ratio to +1 Std Dev Anchored Session VWAP band
86. **`anchored_vwap_dev_lower_1s`**: Distance ratio to -1 Std Dev Anchored Session VWAP band
87. **`anchored_vwap_dev_upper_2s`**: Distance ratio to +2 Std Dev Anchored Session VWAP band
88. **`anchored_vwap_dev_lower_2s`**: Distance ratio to -2 Std Dev Anchored Session VWAP band
89. **`rth_session_vwap_dist`**: Distance ratio to Regular Trading Hours (09:30-16:00 EST) VWAP
90. **`globex_night_vwap_dist`**: Distance ratio to Globex Overnight (18:00-09:30 EST) VWAP
91. **`vwap_stdev2_channel_position`**: Relative location within +2/-2 Std Dev Session VWAP bands
92. **`twap_distance_ratio`**: Stationary distance ratio of price to Time-Weighted Average Price (TWAP)
93. **`dist_session_high`**: Distance ratio to current session high
94. **`dist_session_low`**: Distance ratio to current session low
95. **`dist_prior_session_close`**: Distance ratio to yesterday's settlement close
96. **`dist_prior_session_high`**: Distance ratio to yesterday's session high
97. **`dist_prior_session_low`**: Distance ratio to yesterday's session low
98. **`dist_prior_2day_high`**: Distance ratio to 2-day rolling high
99. **`dist_prior_2day_low`**: Distance ratio to 2-day rolling low
100. **`dist_prior_3day_high`**: Distance ratio to 3-day rolling high
101. **`dist_prior_3day_low`**: Distance ratio to 3-day rolling low
102. **`dist_weekly_open`**: Distance ratio to Weekly Monday 18:00 EST open price
103. **`dist_weekly_high`**: Distance ratio to Developing Weekly High
104. **`dist_weekly_low`**: Distance ratio to Developing Weekly Low
105. **`camarilla_pivot_r3_distance`**: Distance ratio to Camarilla Pivot R3 mean-reversion resistance
106. **`camarilla_pivot_s3_distance`**: Distance ratio to Camarilla Pivot S3 mean-reversion support
107. **`camarilla_pivot_r4_breakout_distance`**: Distance ratio to Camarilla Pivot R4 breakout level
108. **`woodie_pivot_point_distance`**: Distance ratio to Woodie's Daily Pivot Point
109. **`fibonacci_pivot_50_distance`**: Distance ratio to 50% intraday Fibonacci retracement level
110. **`fibonacci_pivot_618_distance`**: Distance ratio to 61.8% Golden Ratio Fibonacci level

---

## Module 4: Volume Delta, Order Flow & Spectral Signals (Features 111 – 145)
111. **`relative_volume_20`**: 20-bar relative volume surge ratio
112. **`relative_volume_60`**: 60-bar relative volume surge ratio
113. **`volume_trend_slope`**: 20-bar volume linear regression slope
114. **`buying_pressure_ratio_15`**: 15-bar volume buying pressure ratio
115. **`selling_pressure_ratio_15`**: 15-bar volume selling pressure ratio
116. **`volume_delta_proxy_15`**: 15-bar net volume delta proxy
117. **`volume_delta_proxy_60`**: 60-bar net volume delta proxy
118. **`cvd_proxy_slope_30`**: 30-bar Cumulative Volume Delta (CVD) proxy linear slope
119. **`volume_weighted_rsi_14`**: 14-bar Volume-Weighted RSI
120. **`chaikin_money_flow_20`**: 20-bar Chaikin Money Flow (CMF)
121. **`money_flow_index_14`**: 14-bar Money Flow Index (MFI)
122. **`ease_of_movement_14`**: 14-bar Ease of Movement indicator
123. **`volume_price_trend_vpt`**: Volume Price Trend (VPT) indicator
124. **`negative_volume_index`**: Negative Volume Index (smart money flow)
125. **`positive_volume_index`**: Positive Volume Index (uninformed volume flow)
126. **`force_index_13`**: 13-bar Alexander Elder Force Index
127. **`force_index_50`**: 50-bar Alexander Elder Force Index
128. **`volume_oscillator_5_20`**: Volume Oscillator difference (5-bar vs 20-bar volume EMA)
129. **`klinger_volume_oscillator`**: Klinger Volume Oscillator
130. **`volume_zone_oscillator`**: Volume Zone Oscillator (VZO) measuring volume flow
131. **`volume_weighted_macd_hist`**: Volume-Weighted MACD histogram difference
132. **`accumulation_distribution_slope_20`**: 20-bar linear regression slope of A/D Line
133. **`intraday_intensity_index`**: Bostian Intraday Intensity Index measuring institutional accumulation
134. **`wiseman_trade_volume_delta`**: Wiseman Wave volume accumulation per price wave swing
135. **`order_imbalance_ratio_5m`**: 5-minute volume imbalance ratio
136. **`order_imbalance_ratio_15m`**: 15-minute volume imbalance ratio
137. **`trades_per_minute_surge`**: Normalized 1-minute trades count surge ratio
138. **`volume_climax_indicator`**: Volume climax spike flag relative to 60-bar ATR
139. **`liquidity_void_distance`**: Distance to recent 1-minute liquidity price gap
140. **`price_density_ratio_30`**: 30-bar price density (total distance traveled vs net displacement)
141. **`market_efficiency_coefficient`**: Perry Kaufman Market Efficiency Coefficient
142. **`fractal_dimension_index`**: Sevcik Fractal Dimension Index over 30 bars
143. **`spectral_density_peak_freq`**: Dominant peak frequency from Fast Fourier Transform (FFT)
144. **`hilbert_phase_angle`**: Instantaneous Hilbert Transform phase angle
145. **`hilbert_quadrature_power`**: Instantaneous Hilbert Transform quadrature power

---

## Module 5: Pairwise Cross-Asset Intermarket Spreads & Correlations (Features 146 – 185)
146. **`nq_es_spread_zscore`**: NQ vs ES normalized price spread z-score
147. **`nq_es_divergence_5m`**: 5-minute return divergence between NQ and ES
148. **`nq_zn_spread_zscore`**: Normalized price spread z-score between NQ and ZN
149. **`nq_cl_spread_zscore`**: Normalized price spread z-score between NQ and CL
150. **`nq_dx_spread_zscore`**: Normalized price spread z-score between NQ and DX
151. **`nq_6e_spread_zscore`**: Normalized price spread z-score between NQ and 6E
152. **`nq_gc_spread_zscore`**: Normalized price spread z-score between NQ and GC
153. **`es_cl_spread_zscore`**: Normalized price spread z-score between ES and CL
154. **`es_dx_spread_zscore`**: Normalized price spread z-score between ES and DX
155. **`es_6e_spread_zscore`**: Normalized price spread z-score between ES and 6E
156. **`es_gc_spread_zscore`**: Normalized price spread z-score between ES and GC
157. **`zn_dx_spread_zscore`**: Normalized price spread z-score between ZN and DX
158. **`zn_6e_spread_zscore`**: Normalized price spread z-score between ZN and 6E
159. **`zn_gc_spread_zscore`**: Normalized price spread z-score between ZN and GC
160. **`cl_dx_spread_zscore`**: Normalized price spread z-score between CL and DX
161. **`cl_gc_spread_zscore`**: Normalized price spread z-score between CL and GC
162. **`dx_6e_spread_zscore`**: Normalized price spread z-score between DX and 6E (Dollar-Euro parity)
163. **`nq_zn_divergence_15m`**: 15-minute return divergence between NQ and ZN
164. **`nq_cl_divergence_15m`**: 15-minute return divergence between NQ and CL
165. **`nq_dx_divergence_15m`**: 15-minute return divergence between NQ and DX
166. **`es_cl_divergence_15m`**: 15-minute return divergence between ES and CL
167. **`es_dx_divergence_15m`**: 15-minute return divergence between ES and DX
168. **`es_gc_divergence_15m`**: 15-minute return divergence between ES and GC
169. **`zn_dx_divergence_15m`**: 15-minute return divergence between ZN and DX
170. **`es_nq_corr_15m`**: 15-minute rolling correlation between ES and NQ
171. **`es_zn_corr_15m`**: 15-minute rolling correlation between ES and ZN
172. **`es_cl_corr_15m`**: 15-minute rolling correlation between ES and CL
173. **`es_dx_corr_15m`**: 15-minute rolling correlation between ES and DX
174. **`nq_zn_corr_60m`**: 60-minute rolling correlation between NQ and ZN
175. **`nq_dx_corr_60m`**: 60-minute rolling correlation between NQ and DX
176. **`zn_dx_corr_60m`**: 60-minute rolling correlation between ZN and DX
177. **`zn_gc_corr_60m`**: 60-minute rolling correlation between ZN and GC
178. **`cl_dx_corr_60m`**: 60-minute rolling correlation between CL and DX
179. **`es_nq_beta_60m`**: 60-minute rolling regression beta of ES against NQ
180. **`es_zn_beta_60m`**: 60-minute rolling regression beta of ES against ZN
181. **`es_dx_beta_60m`**: 60-minute rolling regression beta of ES against DX
182. **`es_cl_beta_60m`**: 60-minute rolling regression beta of ES against CL
183. **`cointeg_spread`**: Cointegration spread residual between ES and NQ
184. **`cointeg_half_life`**: Authentic OLS Ornstein-Uhlenbeck mean-reversion half-life
185. **`cointeg_residual_es_zn`**: Cointegration error residual between ES and ZN

---

## Module 6: Macro Economic Releases, Countdowns & Surprises (Features 186 – 210)
186. **`minutes_to_fomc_decision`**: Minute countdown to scheduled FOMC Interest Rate Decision
187. **`fomc_pre_event_decay_kernel`**: Continuous exponential decay ramp-up kernel before FOMC
188. **`fomc_post_event_memory_kernel`**: Continuous exponential memory decay kernel after FOMC
189. **`fomc_rate_surprise_delta`**: FOMC actual interest rate change vs consensus expectation
190. **`minutes_to_cpi_release`**: Minute countdown to scheduled US CPI Inflation release
191. **`cpi_pre_event_decay_kernel`**: Continuous exponential decay ramp-up kernel before CPI
192. **`cpi_post_event_memory_kernel`**: Continuous exponential memory decay kernel after CPI
193. **`cpi_headline_surprise_zscore`**: Headline CPI YoY actual vs consensus surprise z-score
194. **`cpi_core_surprise_zscore`**: Core CPI YoY actual vs consensus surprise z-score
195. **`minutes_to_ppi_release`**: Minute countdown to scheduled US PPI Producer Price release
196. **`ppi_surprise_zscore`**: PPI YoY actual vs consensus surprise z-score
197. **`minutes_to_pce_release`**: Minute countdown to scheduled Core PCE Inflation release
198. **`pce_core_surprise_zscore`**: Core PCE Inflation actual vs consensus surprise z-score
199. **`minutes_to_nfp_release`**: Minute countdown to scheduled Non-Farm Payrolls release
200. **`nfp_pre_event_decay_kernel`**: Continuous exponential decay ramp-up kernel before NFP
201. **`nfp_post_event_memory_kernel`**: Continuous exponential memory decay kernel after NFP
202. **`nfp_employment_surprise_zscore`**: NFP employment change actual vs consensus surprise z-score
203. **`unemployment_rate_surprise_delta`**: US Unemployment Rate actual vs consensus delta
204. **`hourly_earnings_surprise_zscore`**: Average Hourly Earnings MoM surprise z-score
205. **`minutes_to_jobless_claims`**: Minute countdown to Thursday 08:30 EST Initial Jobless Claims
206. **`jobless_claims_surprise_zscore`**: Initial Jobless Claims actual vs consensus surprise z-score
207. **`minutes_to_ism_pmi_release`**: Minute countdown to scheduled ISM Manufacturing/Services PMI
208. **`ism_mfg_pmi_surprise_zscore`**: ISM Manufacturing PMI actual vs consensus surprise z-score
209. **`michigan_sentiment_surprise`**: University of Michigan Consumer Sentiment surprise z-score
210. **`single_stock_earnings_event_flag`**: Continuous exponential decay kernel for Mega-Cap earnings

---

## Module 7: Real-Time Yield Velocities, Dollar & Risk Regimes (Features 211 – 245)
211. **`zn_10y_yield_proxy`**: 10-Year Treasury Yield proxy derived from `ZN` contract
212. **`zn_yield_velocity_15m`**: 15-minute 10-Year Treasury Yield directional rate of change
213. **`zn_yield_velocity_60m`**: 60-minute 10-Year Treasury Yield directional rate of change
214. **`dx_usd_index_momentum_15m`**: 15-minute US Dollar Index (`DX`) momentum
215. **`dx_usd_index_momentum_60m`**: 60-minute US Dollar Index (`DX`) momentum
216. **`6e_eur_usd_momentum_15m`**: 15-minute EUR/USD (`6E`) momentum
217. **`cl_oil_inflation_impulse_15m`**: 15-minute Crude Oil (`CL`) inflation impulse
218. **`cl_oil_inflation_impulse_60m`**: 60-minute Crude Oil (`CL`) inflation impulse
219. **`gc_gold_safe_haven_impulse_15m`**: 15-minute Gold (`GC`) safe-haven impulse
220. **`gc_gold_safe_haven_impulse_60m`**: 60-minute Gold (`GC`) safe-haven impulse
221. **`risk_on_risk_off_ratio_15m`**: Ratio of NQ 15m return to ZN 15m return (Tech vs Bonds)
222. **`risk_on_risk_off_ratio_60m`**: Ratio of NQ 60m return to ZN 60m return (Tech vs Bonds)
223. **`dollar_equity_divergence_15m`**: Divergence ratio of ES 15m return to DX 15m return
224. **`dollar_equity_divergence_60m`**: Divergence ratio of ES 60m return to DX 60m return
225. **`oil_equity_correlation_impulse`**: 60-minute rolling correlation change between ES and CL
226. **`gold_equity_correlation_impulse`**: 60-minute rolling correlation change between ES and GC
227. **`intraday_implied_vol_proxy_es`**: S&P 500 intraday implied volatility proxy from ES Parkinson vol
228. **`intraday_implied_vol_proxy_nq`**: Nasdaq 100 intraday implied volatility proxy from NQ Parkinson vol
229. **`volatility_term_structure_ratio`**: Ratio of 5-minute implied vol proxy to 60-minute vol proxy
230. **`vix_spike_flag_proxy`**: Binary/continuous flag when ES Parkinson vol exceeds 3.0 z-scores
231. **`credit_spread_proxy_zn_dx`**: Interest rate credit/liquidity risk proxy (ZN vs DX return ratio)
232. **`fed_hawkish_dovish_score`**: Real-time composite score of ZN yield velocity and DX momentum
233. **`macro_liquidity_impulse`**: Composite sum of 6E, DX, ZN, and CL momentum impulses
234. **`inflation_expectations_proxy`**: Composite ratio of CL crude oil and GC gold returns to ZN notes
235. **`cross_asset_volatility_dispersion`**: Standard deviation of rolling realized vol across all 7 assets
236. **`cross_asset_momentum_dispersion`**: Standard deviation of 15m returns across all 7 assets
237. **`market_breadth_proxy_7asset`**: Fraction of the 7 futures assets exhibiting positive 15m return
238. **`kalman_price_dev`**: Stationary price deviation from Kalman filter trend
239. **`kalman_trend_velocity`**: Velocity of Kalman filter trend line
240. **`kalman_innovation_zscore`**: Z-score of Kalman filter innovation residual
241. **`kalman_uncertainty`**: State variance of Kalman filter error covariance
242. **`hmm_prob_regime_0`**: Gaussian HMM probability for Regime 0 (Trending)
243. **`hmm_prob_regime_1`**: Gaussian HMM probability for Regime 1 (Mean-Reverting)
244. **`hmm_prob_regime_2`**: Gaussian HMM probability for Regime 2 (High-Vol Spillover)
245. **`macro_regime_composite_index`**: Overall 3-state macro risk regime index (Risk-On / Neutral / Panic)

---

## Module 8: Underlying Constituent Dynamics, Market Breadth & Index Arbitrage Basis (Features 246 – 280)
246. **`megacap7_weighted_momentum_15m`**: Weighted 15-minute return impulse of top 7 mega-caps (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA)
247. **`megacap_vs_equalweight_divergence`**: Return spread between Cap-Weighted Index (`ES`) and Equal-Weighted S&P 500 (`RSP` proxy)
248. **`constituent_concentration_index`**: Realized Herfindahl-Hirschman index of constituent return contributions
249. **`megacap_volatility_spillover_60m`**: Realized volatility spillover ratio from top 7 mega-caps into ES futures
250. **`megacap_order_flow_imbalance`**: Composite signed volume delta ratio across top 7 mega-cap equities
251. **`advance_decline_line_velocity`**: 15-minute rate of change of the S&P 500 Advance-Decline (A/D) volume ratio
252. **`up_down_volume_imbalance_ratio`**: Ratio of buying volume in advancing constituent stocks vs selling volume in declining stocks
253. **`new_highs_new_lows_ratio_15m`**: Fraction of index constituents making 15-minute rolling new highs vs new lows
254. **`breadth_momentum_oscillator`**: 14-bar momentum oscillator of advancing constituent volume
255. **`trin_arms_index_proxy`**: Arms Index (TRIN) proxy measuring market breadth congestion
256. **`cash_futures_fair_value_basis`**: Continuous intraday price spread between `ES` futures and theoretical synthetic cash index ($S_t - F_t$)
257. **`index_arb_mispricing_zscore`**: Z-score of cash-futures basis (signals imminent HFT arbitrage basket execution)
258. **`etf_creation_redemption_imbalance`**: Volume delta proxy in SPY/QQQ ETF creation vs redemption baskets
259. **`cash_futures_lead_lag_correlation`**: 15-minute rolling correlation between synthetic cash index returns and ES futures returns
260. **`basis_momentum_impulse`**: Rate of change of the cash-futures basis spread over 5 bars
261. **`implied_correlation_index_dispersion`**: Variance difference between individual constituent volatility and index volatility
262. **`sector_rotation_lead_lag_beta`**: Tech (`XLK`) vs Financials (`XLF`) vs Energy (`XLE`) relative strength momentum differential
263. **`cross_sector_momentum_dispersion`**: Standard deviation of 15m returns across 11 S&P 500 GICS sector proxies
264. **`synthetic_underlying_cash_momentum`**: Synthetic 1-minute return momentum of reconstructed cash index basket
265. **`value_area_high_distance`**: Distance ratio to Developing Session Value Area High (VAH)
266. **`value_area_low_distance`**: Distance ratio to Developing Session Value Area Low (VAL)
267. **`point_of_control_distance`**: Distance ratio to Developing Session Point of Control (POC)
268. **`market_profile_balance_ratio`**: Value Area width to Session Range width ratio
269. **`nq_close_scaled`**: Stationary scaled NQ close return
270. **`nq_return_5m`**: NQ 5-minute return
271. **`zn_return_5m`**: 10-Yr Treasury Note (`ZN`) 5-minute return
272. **`cl_return_5m`**: Crude Oil (`CL`) 5-minute return
273. **`dx_return_5m`**: US Dollar Index (`DX`) 5-minute return
274. **`6e_return_5m`**: EUR/USD (`6E`) 5-minute return
275. **`gc_return_5m`**: Gold (`GC`) 5-minute return
276. **`es_zn_corr_60m`**: 60-minute rolling correlation between ES and ZN
277. **`es_cl_corr_60m`**: 60-minute rolling correlation between ES and CL
278. **`es_dx_corr_60m`**: 60-minute rolling correlation between ES and DX
279. **`es_6e_corr_60m`**: 60-minute rolling correlation between ES and 6E
280. **`es_gc_corr_60m`**: 60-minute rolling correlation between ES and GC

---

## Module 9: Advanced Spectral, Physics & Non-Linear Dynamics (Features 281 – 315)
281. **`cwt_morlet_scalogram_octave_ratio`**: Continuous Wavelet Transform (CWT) Morlet energy scalogram ratio across 4 octave scales
282. **`ewt_adaptive_noise_subband_var`**: Empirical Wavelet Transform (EWT) sub-band variance isolating high-frequency noise modes
283. **`swt_shift_invariant_detail_coef`**: Stationary Wavelet Transform (SWT) shift-invariant detail coefficient
284. **`dwt_energy_subband_1`**: Discrete Wavelet Transform high-frequency sub-band 1 energy
285. **`dwt_energy_subband_2`**: Discrete Wavelet Transform medium-frequency sub-band 2 energy
286. **`dwt_energy_subband_3`**: Discrete Wavelet Transform low-frequency sub-band 3 energy
287. **`emd_imf1_instantaneous_freq`**: Empirical Mode Decomposition (EMD) Intrinsic Mode Function 1 (IMF 1) instantaneous frequency
288. **`emd_imf2_instantaneous_phase_diff`**: EMD IMF 2 instantaneous phase difference between price velocity and volume velocity
289. **`takens_phase_space_curvature`**: 3D Phase Space attractor trajectory curvature reconstructed via Takens' delay coordinate embedding
290. **`tda_1d_persistence_loop_score`**: Topological Data Analysis (TDA) 1D persistent homology score measuring range-bound orbits vs breakout manifolds
291. **`largest_lyapunov_exponent_chaoticity`**: Trailing Largest Lyapunov Exponent (LLE) measuring exponential divergence of phase-space trajectories
292. **`mf_dfa_multifractal_spectrum_width`**: Multifractal Detrended Fluctuation Analysis (MF-DFA) spectrum width ($\Delta \alpha$)
293. **`tsallis_non_extensive_entropy`**: Tsallis generalized statistical entropy ($q=1.5$) tailored for non-Gaussian fat-tailed return distributions
294. **`shannon_entropy_60`**: 60-bar return distribution Shannon Entropy (disorder score)
295. **`lempel_ziv_complexity_100`**: Lempel-Ziv binary sequence complexity score
296. **`schrodinger_psi_squared_prob`**: Probability density $|\psi(x,t)|^2$ of price trapped in a quantum harmonic potential well around VWAP
297. **`quantum_tunneling_breakout_prob`**: Probability of price tunneling through a high-volume resistance potential barrier $V(x)$
298. **`boltzmann_gibbs_thermo_entropy`**: Statistical mechanics thermodynamic entropy of 1-minute return energy states
299. **`bose_einstein_condensation_score`**: Degree of spectral concentration across lower frequency modes
300. **`fermi_dirac_liquidity_occupancy`**: Occupancy probability of liquidity price levels based on Fermi-Dirac distribution
301. **`feynman_path_integral_action`**: Least Action Principle score $\mathcal{S} = \int (T - V) dt$ along price trajectories
302. **`quantum_hamiltonian_eigenvalue`**: Energy eigenvalue score of the price operator Hamiltonian $\hat{H} = -\frac{\hbar^2}{2m} \nabla^2 + V(x)$
303. **`phase_coherence_index`**: Cross-asset phase angle alignment across `ES`, `NQ`, and `ZN`
304. **`helmholtz_free_energy_proxy`**: Helmholtz Free Energy proxy ($F = U - TS$) measuring available kinetic trading energy vs entropy
305. **`percolation_cluster_size`**: Critical percolation threshold of price connectivity across intraday support/resistance nodes
306. **`transfer_entropy_zn_to_es`**: Directional information flow from Treasury Notes (`ZN`) to S&P 500 (`ES`) via Transfer Entropy $TE(ZN \to ES)$
307. **`transfer_entropy_dx_to_es`**: Directional information flow from US Dollar (`DX`) to S&P 500 (`ES`)
308. **`transfer_entropy_cl_to_es`**: Directional information flow from Crude Oil (`CL`) to S&P 500 (`ES`)
309. **`granger_causality_fstat_nq_es`**: Rolling Granger Causality F-statistic from NQ to ES
310. **`symbolic_transfer_entropy_rates`**: Discretized ordinal pattern information flow rate between interest rates and equities
311. **`active_information_storage`**: Amount of information stored in past price memory available for next-bar prediction
312. **`excess_entropy_complexity`**: Total mutual information between past and future price series
313. **`permutation_entropy_normalized`**: Bandt-Pompe Permutation Entropy measuring price pattern randomness
314. **`directed_info_flow_asymmetry`**: Net information flow asymmetry ratio $\frac{TE(A \to B)}{TE(B \to A)}$
315. **`mutual_info_rate_of_change`**: Rate of change of mutual information between volume flow and price returns

---

## Module 10: Market Microstructure, Transfer Entropy & Copula Tail Risk (Features 316 – 350)
316. **`bid_ask_spread_corwin_schultz`**: Corwin-Schultz high-frequency bid-ask spread estimator derived from high/low ratios
317. **`roll_implicit_bid_ask_spread`**: Roll (1984) implicit bid-ask spread proxy calculated from return serial covariance
318. **`trade_size_dispersion_ratio`**: Ratio of high-volume bar frequency to total trade activity (institutional block order proxy)
319. **`kyles_lambda_price_impact`**: Kyle's Lambda price impact coefficient measuring price change per unit signed volume
320. **`hasbrouck_information_share`**: Price discovery contribution ratio derived from cross-asset vector autoregression (VAR) residuals
321. **`vpin_volume_sync_pin`**: Volume-Synchronized Probability of Informed Trading (VPIN) toxicity metric
322. **`order_flow_toxicity_zscore`**: Z-score of rapid directional volume imbalance bursts
323. **`quote_stuffing_surge_indicator`**: Burst indicator for micro-second bar trade frequency anomalies
324. **`tick_rule_signed_momentum`**: Lee-Ready tick rule signed volume momentum proxy
325. **`microstructure_noise_variance`**: Bandpass-filtered microstructure noise variance vs underlying price diffusion variance
326. **`volatility_stop_distance`**: Distance ratio to Parabolic SAR / Volatility Stop trailing line
327. **`copula_lower_tail_dependence`**: Extreme downside bivariate copula tail dependence coefficient between ES and NQ
328. **`copula_upper_tail_dependence`**: Extreme upside bivariate copula tail dependence coefficient between ES and NQ
329. **`cvar_95_expected_shortfall`**: 95% Conditional Value-at-Risk (CVaR) tail loss proxy
330. **`evt_gev_shape_parameter`**: Generalized Extreme Value (GEV) distribution shape parameter $\xi$ (tail heaviness)
331. **`realized_semivariance_ratio`**: Ratio of positive realized variance to negative realized variance ($\frac{RV^+}{RV^-}$)
332. **`jump_bipower_variation_ratio`**: Barattieri-Barndorff-Nielsen-Shephard ratio of Bipower Variation to Realized Variance
333. **`higher_order_moment_asymmetry`**: Higher-order moment return asymmetry score
334. **`markov_switching_regime_prob`**: 2-state Markov-Switching Autoregressive probability of high-variance regime
335. **`cantelli_max_drawdown_zscore`**: Z-score of potential maximum drawdown based on Cantelli's inequality
336. **`evt_hill_tail_index`**: Hill estimator for heavy-tail exponent $\alpha$ of negative return extremes
337. **`amihud_illiquidity_20`**: 20-bar Amihud illiquidity ratio
338. **`volume_force`**: Product of log return and volume
339. **`cointeg_residual_es_dx`**: Cointegration error residual between ES and DX
340. **`cointeg_residual_es_gc`**: Cointegration error residual between ES and GC
341. **`cointeg_residual_nq_zn`**: Cointegration error residual between NQ and ZN
342. **`cointeg_residual_cl_dx`**: Cointegration error residual between CL and DX
343. **`relative_strength_es_nq`**: 60-minute relative performance ratio ES / NQ
344. **`relative_strength_es_zn`**: 60-minute relative performance ratio ES / ZN
345. **`relative_strength_es_dx`**: 60-minute relative performance ratio ES / DX
346. **`relative_strength_nq_gc`**: 60-minute relative performance ratio NQ / GC
347. **`relative_strength_cl_dx`**: 60-minute relative performance ratio CL / DX
348. **`dist_prior_session_close`**: Distance ratio to prior day settlement close
349. **`dist_prior_session_high`**: Distance ratio to prior day session high
350. **`dist_prior_session_low`**: Distance ratio to prior day session low
