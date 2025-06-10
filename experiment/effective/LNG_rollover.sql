PATTERN SEQ(A+, B+, C)
SEMANTIC skip_till_any_match
WHERE A.type = 'boil_off_rate'
AND standard_deviation(A) / avg(A) >= 1.1 -- 蒸发率急剧变化
AND B.type = 'density_difference'
AND B[i].value - B[i-1].value < 0 -- 密度差下降
AND C.type = 'tank_pressure'
AND C.value >= 250 -- 压力超过罐柜安全阈值
WITHIN max_time


