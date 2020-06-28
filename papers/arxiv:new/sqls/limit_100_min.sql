SELECT depth, MIN(RES), MAX(RES) FROM
-- SELECT * FROM
(
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=5 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=10 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=15 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=20 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=25 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=30 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=35 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=40 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
UNION
SELECT * FROM
	(SELECT depth,  res_depth  RES FROM "9_training_no_analysis" WHERE depth=45 AND res_depth != -1 ORDER BY RES ASC LIMIT 10000)
)
GROUP BY depth

