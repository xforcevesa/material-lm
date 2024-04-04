for dir in $(ls)
do
    cd $dir
    cat wget-list.txt | parallel --gnu "wget {}"
    cd -
done
