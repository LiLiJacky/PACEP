PATTERN SEQ(a, b+, c)
SEMATIC skip_till_any_match
WHERE a.type = 'reducer_start'
AND b.type = 'load_std'
AND b[i].value - b[i-1].value >= 0
AND c.type = 'reducer_end'
WITHIN 10 minutes

