#!/bin/sh
#python3 colision_avoidance_net_test_parse.py --num_nodes {} {} {}
for i in 20 40 60 80
do
	for j in 20 40 60 80
	do
		for k in 20 40 60 80
		do
			echo "Start train node ${i}_${j}_${k}" 
			python3 colision_avoidance_net.py --num_nodes "${i}" "${j}" "${k}"
#		echo "Start train node ${i}_${j}_40"
#               python3 colision_avoidance_net_test_parse.py --num_nodes "${i}" "${j}" 40 &
#		echo "Start train node ${i}_${j}_60"
#               python3 colision_avoidance_net_test_parse.py --num_nodes "${i}" "${j}" 60 &
#		echo "Start train node ${i}_${j}_80"
#               python3 colision_avoidance_net_test_parse.py --num_nodes "${i}" "${j}" 80
		done
	done
done
