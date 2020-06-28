-- Replace the fields to select in the topmost query and in the last lime that 
-- groups the selected data. For example, SELECT depth, cx, COUNT(*).
-- If the criteria to order should include time, then avg(res_depth) 
-- should be replaced everywhere with avg(res_depth * init_time)

SELECT depth, att_c, COUNT(*) FROM
(
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 5 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 10 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 15 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 20 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 25 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 30 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 35 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 40 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
UNION
SELECT * FROM
(SELECT *, avg(res_depth) as ARD FROM "9_training_no_analysis" 
	WHERE depth = 45 AND res_depth != -1
	GROUP BY max_depth, max_children, att_b, att_c, div_dist, cx, depth
	ORDER BY ARD ASC
	LIMIT 100
)
)
GROUP BY depth, att_c