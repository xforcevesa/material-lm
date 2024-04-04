# shellcheck disable=SC2045
for dir in $(ls)
do
    cd "$dir" || exit
    wget -N -i wget-list.txt
    cd - || exit
done
