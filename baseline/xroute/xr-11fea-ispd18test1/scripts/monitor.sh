while true; do
    for file in logs/worker*.log; do
        echo -n "[$file] "
        tail -n 1 "$file"
    done
    echo ""
    echo ""
    sleep 10
done
