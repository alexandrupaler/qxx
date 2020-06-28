--- To calculate the number of timeouts:
--- change the parameter e.g. 20 in all the subsequence statements
--- res_depth = -1 means that a particular simulation timedout during simulation
--- checking init_time > 20 is checking 'how many simulations have time larger than 20 seconds?'

SELECT 9, max_children, COUNT(*) as ARD FROM "9_training_no_analysis" 
	WHERE res_depth = -1 OR init_time > 20
	GROUP BY max_children
UNION
SELECT 5, max_children, COUNT(*) as ARD FROM "5_training_no_analysis" 
	WHERE res_depth = -1 OR init_time > 20
	GROUP BY max_children
UNION
SELECT 1, max_children, COUNT(*) as ARD FROM "1_training_no_analysis" 
	WHERE res_depth = -1 OR init_time > 20
	GROUP BY max_children