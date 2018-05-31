#write wires
for i in `seq 0 $(($1 - 1))`;
do
    echo "q$i W"
done

IFS=$'\n'
cxs=( $(grep cx $2) )
for cx in "${cxs[@]}"
do
#    echo $cx
    pl0="${cx/cx /}"
    pl1="${pl0//[\]\[;]/}"
    pl2="${pl1//[,]/ +}"
    echo $pl2
    echo "TOUCH"
done
