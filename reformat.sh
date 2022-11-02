#!/bin/bash

# limiting to 2000 examples per class
main(){
    n1=0
    n2=0
    n3=0
    n4=0
    limit=2000
    
    for file in *; do
        if [[ "$file" == *.jpg || "$file" == *.JPG ]]; then
            echo "$limit $n1 $n2 $n3 $n4"
            n=$((n1+n2+n3+n4))
            if [[ "$file" == HAZE* && "$n1" -lt "$limit" ]]; then
                echo "1" > "../MWI-reformat/labels/$n.txt"
                cp $file "../MWI-reformat/data/$n.jpg"
                n1=$((n1+1))
            fi
            if [[ "$file" == RAINY* && "$n2" -lt "$limit" ]]; then
                echo "2" > "../MWI-reformat/labels/$n.txt"
                cp $file "../MWI-reformat/data/$n.jpg"
                n2=$((n2+1))
            fi
            if [[ "$file" == SNOWY* && "$n3" -lt "$limit" ]]; then
                echo "3" > "../MWI-reformat/labels/$n.txt"
                cp $file "../MWI-reformat/data/$n.jpg"
                n3=$((n3+1))
            fi
            if [[ "$file" == SUNNY* && "$n4" -lt "$limit" ]]; then
                echo "4" > "../MWI-reformat/labels/$n.txt"
                cp $file "../MWI-reformat/data/$n.jpg"
                n4=$((n4+1))
            fi
        fi
    done
}

cd "../data" && main